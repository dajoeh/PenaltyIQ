import os
import math
import sqlite3
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ==============================================================================
# STEP 4 PRODUCTION CONFIGURATION
# Points directly to the Pillar 7 SQLite database exported from your notebook
# ==============================================================================
DEFAULT_PROD_DB = os.path.join("output_data", "epl_synthetic_penalties.db")
DB_PATH = os.environ.get(
    "PENALTY_DB_PATH",
    DEFAULT_PROD_DB if os.path.exists(DEFAULT_PROD_DB) else "dummy_penalties.db",
)
H2H_PRIOR_WEIGHT_KAPPA = 4.0  # Weight of indirect log-odds prior vs direct H2H stops


def get_db_connection():
    """Opens a SQLite connection with row_factory set to sqlite3.Row."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database file not found at '{DB_PATH}'. "
            f"Ensure 'output_data/epl_synthetic_penalties.db' exists in your project directory."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _logit(p: float) -> float:
    p_clipped = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p_clipped / (1.0 - p_clipped))


def _inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def get_league_baselines(conn: sqlite3.Connection) -> dict:
    """
    Computes empirical league-wide penalty baselines directly from the database.
    """
    row = conn.execute(
        """
        SELECT
            SUM(Total_PK_Kicked) AS total_kicks,
            SUM(Converted) AS total_conv,
            SUM(Stopped_by_GK) AS total_stop,
            SUM(Missed_Off_Target) AS total_miss
        FROM dataset_1_kickers
        """
    ).fetchone()

    total_kicks = int(row["total_kicks"] or 0)
    if total_kicks < 100:
        return {
            "total_penalties_in_db": total_kicks,
            "league_p_on_target": 0.9250,
            "league_p_stop_given_target": 0.1500,
            "league_p_conv_given_target": 0.8500,
            "league_p_conv_overall": 0.7863,
            "league_p_stop_overall": 0.1388,
        }

    total_conv = int(row["total_conv"] or 0)
    total_stop = int(row["total_stop"] or 0)
    total_on_target = max(1, total_conv + total_stop)

    p_on_target = total_on_target / total_kicks
    p_stop_given_target = total_stop / total_on_target
    p_conv_given_target = total_conv / total_on_target

    return {
        "total_penalties_in_db": total_kicks,
        "league_p_on_target": round(p_on_target, 4),
        "league_p_stop_given_target": round(p_stop_given_target, 4),
        "league_p_conv_given_target": round(p_conv_given_target, 4),
        "league_p_conv_overall": round(total_conv / total_kicks, 4),
        "league_p_stop_overall": round(total_stop / total_kicks, 4),
    }


def get_featured_test_pairs(conn: sqlite3.Connection) -> dict:
    """
    Dynamically discovers 3 real test pairs from the connected database:
    1. Repeat Rivalry (highest H2H encounter count in dataset_3_matchup_events)
    2. First-Time Encounter (high-volume kicker & GK with 0 prior meetings)
    3. Small-Sample Kicker (kicker with exactly 1 career PK vs top GK)
    """
    # 1. Top Repeat Rivalry Pair
    repeat_row = conn.execute(
        """
        SELECT kicker_number, gk_number, COUNT(*) AS meetings
        FROM dataset_3_matchup_events
        GROUP BY kicker_number, gk_number
        ORDER BY meetings DESC, SUM(CASE WHEN outcome = 'Stopped_by_GK' THEN 1 ELSE 0 END) DESC
        LIMIT 1
        """
    ).fetchone()

    # 2. Top First-Time Encounter Pair (n_ij = 0 among high-volume players)
    first_time_row = conn.execute(
        """
        SELECT k.player_number AS kicker_number, g.player_number AS gk_number
        FROM (
            SELECT player_number, Total_PK_Kicked
            FROM dataset_1_kickers
            ORDER BY Total_PK_Kicked DESC
            LIMIT 15
        ) k
        CROSS JOIN (
            SELECT player_number, Total_PK_Faced
            FROM dataset_2_goalkeepers
            ORDER BY Total_PK_Faced DESC
            LIMIT 15
        ) g
        LEFT JOIN dataset_3_matchup_events e
            ON e.kicker_number = k.player_number AND e.gk_number = g.player_number
        WHERE e.event_id IS NULL
        ORDER BY (k.Total_PK_Kicked + g.Total_PK_Faced) DESC
        LIMIT 1
        """
    ).fetchone()

    # 3. Small-Sample Kicker (1 career attempt) vs Top GK
    small_k_row = conn.execute(
        """
        SELECT player_number FROM dataset_1_kickers
        WHERE Total_PK_Kicked = 1
        ORDER BY Converted DESC, player_number ASC
        LIMIT 1
        """
    ).fetchone()
    top_gk_row = conn.execute(
        "SELECT player_number FROM dataset_2_goalkeepers ORDER BY Total_PK_Faced DESC LIMIT 1"
    ).fetchone()

    return {
        "repeat_rivalry": {
            "kicker_number": int(repeat_row["kicker_number"]) if repeat_row else 10,
            "gk_number": int(repeat_row["gk_number"]) if repeat_row else 1,
            "meetings": int(repeat_row["meetings"]) if repeat_row else 0,
        },
        "first_time": {
            "kicker_number": int(first_time_row["kicker_number"]) if first_time_row else 11,
            "gk_number": int(first_time_row["gk_number"]) if first_time_row else 2,
        },
        "small_sample": {
            "kicker_number": int(small_k_row["player_number"]) if small_k_row else 12,
            "gk_number": int(top_gk_row["player_number"]) if top_gk_row else 1,
        },
    }


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Serves the main HTML dashboard."""
    return render_template("index.html")


@app.route("/api/directory", methods=["GET"])
def api_directory():
    """
    Returns all kickers, goalkeepers, league baselines, and dynamically
    discovered featured matchup pairs from epl_synthetic_penalties.db.
    """
    try:
        with get_db_connection() as conn:
            kickers = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT player_number, player_name, primary_squad_number,
                           tenure_category, clubs_played_for, Total_PK_Kicked,
                           Converted, Stopped_by_GK, Missed_Off_Target, eb_prob_convert
                    FROM dataset_1_kickers
                    ORDER BY Total_PK_Kicked DESC, player_number ASC
                    """
                ).fetchall()
            ]
            goalkeepers = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT player_number, player_name, primary_squad_number,
                           tenure_category, clubs_played_for, Total_PK_Faced,
                           Stopped_by_GK, Converted_by_Kicker,
                           Missed_Wide_or_Post_by_Kicker, eb_prob_stop_overall
                    FROM dataset_2_goalkeepers
                    ORDER BY Total_PK_Faced DESC, player_number ASC
                    """
                ).fetchall()
            ]
            baselines = get_league_baselines(conn)
            featured_pairs = get_featured_test_pairs(conn)

        return jsonify({
            "status": "ok",
            "db_path": DB_PATH,
            "total_kickers": len(kickers),
            "total_goalkeepers": len(goalkeepers),
            "league_baselines": baselines,
            "featured_pairs": featured_pairs,
            "kickers": kickers,
            "goalkeepers": goalkeepers,
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/matchup", methods=["GET"])
def api_matchup():
    """
    Computes and returns the cross-probabilities and historical matchup log
    for Kicker # (`kicker_number`) vs Goalkeeper # (`gk_number`).
    """
    kicker_param = request.args.get("kicker_number", "").strip()
    gk_param = request.args.get("gk_number", "").strip()

    if not kicker_param.isdigit() or not gk_param.isdigit():
        return jsonify({
            "status": "error",
            "message": "Please enter valid integer numbers for both Kicker # and Goalkeeper #.",
        }), 400

    k_num = int(kicker_param)
    g_num = int(gk_param)

    try:
        with get_db_connection() as conn:
            k_row = conn.execute(
                "SELECT * FROM dataset_1_kickers WHERE player_number = ?", (k_num,)
            ).fetchone()

            if k_row is None:
                is_gk = conn.execute(
                    "SELECT player_name FROM dataset_2_goalkeepers WHERE player_number = ?",
                    (k_num,),
                ).fetchone()
                if is_gk:
                    return jsonify({
                        "status": "error",
                        "message": f"Player #{k_num} ({is_gk['player_name']}) is a Goalkeeper, not an Outfield Kicker.",
                    }), 404
                return jsonify({
                    "status": "error",
                    "message": f"Kicker #{k_num} was not found in dataset_1_kickers.",
                }), 404

            g_row = conn.execute(
                "SELECT * FROM dataset_2_goalkeepers WHERE player_number = ?", (g_num,)
            ).fetchone()

            if g_row is None:
                is_kicker = conn.execute(
                    "SELECT player_name FROM dataset_1_kickers WHERE player_number = ?",
                    (g_num,),
                ).fetchone()
                if is_kicker:
                    return jsonify({
                        "status": "error",
                        "message": f"Player #{g_num} ({is_kicker['player_name']}) is an Outfield Kicker, not a Goalkeeper.",
                    }), 404
                return jsonify({
                    "status": "error",
                    "message": f"Goalkeeper #{g_num} was not found in dataset_2_goalkeepers.",
                }), 404

            kicker = dict(k_row)
            gk = dict(g_row)
            baselines = get_league_baselines(conn)

            h2h_rows = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT event_id, season, half_season, date, minute,
                           kicker_number, kicker_squad_number, kicker_name, kicker_club,
                           gk_number, gk_squad_number, gk_name, gk_club, outcome
                    FROM dataset_3_matchup_events
                    WHERE kicker_number = ? AND gk_number = ?
                    ORDER BY date DESC, minute DESC
                    """,
                    (k_num, g_num),
                ).fetchall()
            ]

        # ----------------------------------------------------------------------
        # PILLAR 4 CROSS-PROBABILITY CALCULATION
        # ----------------------------------------------------------------------
        p_target_i = float(kicker["eb_prob_on_target"])
        p_conv_target_i = float(kicker["eb_prob_conv_given_target"])
        p_stop_target_j = float(gk["eb_prob_stop_given_target"])
        league_stop_target = float(baselines["league_p_stop_given_target"])

        k_stop_vuln_i = 1.0 - p_conv_target_i
        logit_indirect_stop = (
            _logit(p_stop_target_j)
            + _logit(k_stop_vuln_i)
            - _logit(league_stop_target)
        )
        p_stop_target_indirect = _inv_logit(logit_indirect_stop)

        n_h2h_total = len(h2h_rows)
        n_h2h_conv = sum(1 for r in h2h_rows if r["outcome"] == "Converted")
        n_h2h_stop = sum(1 for r in h2h_rows if r["outcome"] == "Stopped_by_GK")
        n_h2h_miss = sum(1 for r in h2h_rows if r["outcome"] == "Missed_Off_Target")
        n_h2h_on_target = n_h2h_conv + n_h2h_stop

        p_stop_target_combined = (
            (H2H_PRIOR_WEIGHT_KAPPA * p_stop_target_indirect + n_h2h_stop)
            / (H2H_PRIOR_WEIGHT_KAPPA + n_h2h_on_target)
        )
        h2h_weight_share = n_h2h_on_target / (H2H_PRIOR_WEIGHT_KAPPA + n_h2h_on_target)

        cross_prob_missed = 1.0 - p_target_i
        cross_prob_stopped = p_target_i * p_stop_target_combined
        cross_prob_convert = p_target_i * (1.0 - p_stop_target_combined)

        return jsonify({
            "status": "ok",
            "db_path": DB_PATH,
            "kicker": kicker,
            "goalkeeper": gk,
            "league_baselines": baselines,
            "h2h_summary": {
                "encounter_scenario": (
                    "First_Time_Encounter" if n_h2h_total == 0 else "Repeat_Encounter"
                ),
                "prior_h2h_total": n_h2h_total,
                "prior_h2h_on_target": n_h2h_on_target,
                "prior_h2h_converted": n_h2h_conv,
                "prior_h2h_stopped_by_gk": n_h2h_stop,
                "prior_h2h_missed_off_target": n_h2h_miss,
                "h2h_evidence_weight": round(h2h_weight_share, 4),
                "indirect_prior_weight": round(1.0 - h2h_weight_share, 4),
            },
            "matchup_probabilities": {
                "indirect_on_target_stop_prob": round(p_stop_target_indirect, 4),
                "combined_on_target_stop_prob": round(p_stop_target_combined, 4),
                "cross_prob_convert": round(cross_prob_convert, 4),
                "cross_prob_stopped_by_gk": round(cross_prob_stopped, 4),
                "cross_prob_missed_off_target": round(cross_prob_missed, 4),
            },
            "h2h_events": h2h_rows,
        })

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    print(f"Starting Penalty Matchup Server connected to: {DB_PATH}")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)