import os
import numpy as np
import joblib
import tensorflow as tf
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model   = tf.keras.models.load_model(os.path.join(BASE_DIR, "churn.keras"))
scaler  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


def predict_churn(sessions_per_week, avg_session_duration, player_level, achievements_unlocked):
    arr    = np.array([[sessions_per_week, avg_session_duration,
                        player_level, achievements_unlocked]], dtype=np.float32)
    scaled = scaler.transform(arr)
    prob   = float(model.predict(scaled, verbose=0)[0][0])

    bar_filled = int(prob * 30)
    bar        = "█" * bar_filled + "░" * (30 - bar_filled)

    if prob > 0.70:
        risk  = "🔴  HIGH RISK"
        color = "danger"
        action = (
            "Immediate action required:\n"
            "  • Offer a personalised discount or exclusive in-game reward\n"
            "  • Trigger a win-back push notification campaign\n"
            "  • Consider a direct outreach from the support team"
        )
    elif prob > 0.40:
        risk  = "🟡  MEDIUM RISK"
        color = "warning"
        action = (
            "Preventive action recommended:\n"
            "  • Send personalised daily missions and challenges\n"
            "  • Enable push reminders for incomplete tasks\n"
            "  • Introduce a limited-time seasonal event to re-engage"
        )
    else:
        risk  = "🟢  LOW RISK"
        color = "success"
        action = (
            "Player is healthy — keep the momentum:\n"
            "  • Reward with loyalty points or exclusive cosmetics\n"
            "  • Unlock a new achievement tier to maintain motivation\n"
            "  • Feature them on in-game leaderboards"
        )

    result = (
        f"{'─' * 50}\n"
        f"  CHURN RISK PROBABILITY\n"
        f"{'─' * 50}\n"
        f"  {bar}\n"
        f"  Probability : {prob:.1%}  ({prob:.4f})\n"
        f"  Assessment  : {risk}\n"
        f"{'─' * 50}\n\n"
        f"  PLAYER PROFILE\n"
        f"{'─' * 50}\n"
        f"  Sessions / week          : {sessions_per_week}\n"
        f"  Avg session duration     : {avg_session_duration} min\n"
        f"  Player level             : {player_level}\n"
        f"  Achievements unlocked    : {achievements_unlocked}\n"
        f"{'─' * 50}\n\n"
        f"  RECOMMENDED ACTION\n"
        f"{'─' * 50}\n"
        f"  {action}\n"
        f"{'─' * 50}"
    )

    return round(prob, 4), result


with gr.Blocks(title="🎮 Player Churn Risk Prediction", theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
        # 🎮 Player Churn Risk Prediction
        **Deep Neural Network** trained on **100,034** player records  
        `Accuracy: 86.24%` &nbsp;|&nbsp; `AUC-ROC: 0.885` &nbsp;|&nbsp; `4 selected features via ANOVA F-test`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Player Inputs")

            sessions = gr.Slider(
                minimum=0, maximum=19, value=5, step=1,
                label="Sessions Per Week",
                info="How many times the player logs in per week"
            )
            duration = gr.Slider(
                minimum=10, maximum=179, value=60, step=1,
                label="Avg Session Duration (minutes)",
                info="Average length of each gaming session"
            )
            level = gr.Slider(
                minimum=1, maximum=99, value=30, step=1,
                label="Player Level",
                info="In-game progression level (1–99)"
            )
            achievements = gr.Slider(
                minimum=0, maximum=49, value=15, step=1,
                label="Achievements Unlocked",
                info="Total achievements earned (0–49)"
            )

            predict_btn = gr.Button("🔍 Predict Churn Risk", variant="primary", size="lg")

            gr.Markdown("#### Quick Examples")
            with gr.Row():
                ex1 = gr.Button("At-risk player",  size="sm")
                ex2 = gr.Button("Average player",  size="sm")
                ex3 = gr.Button("Loyal player",    size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### Prediction Result")
            prob_out   = gr.Number(label="Churn Probability (0 = safe, 1 = certain churn)")
            result_out = gr.Textbox(label="Full Risk Report", lines=22, max_lines=22)

    predict_btn.click(
        fn=predict_churn,
        inputs=[sessions, duration, level, achievements],
        outputs=[prob_out, result_out]
    )

    ex1.click(fn=lambda: (2,  15,  5,  5),  outputs=[sessions, duration, level, achievements])
    ex2.click(fn=lambda: (10, 90,  40, 20), outputs=[sessions, duration, level, achievements])
    ex3.click(fn=lambda: (18, 160, 80, 45), outputs=[sessions, duration, level, achievements])

    gr.Markdown(
        """
        ---
        **Model info:** Deep Neural Network — 256→128→64→32 neurons, BatchNormalization, Dropout  
        **Features selected:** SessionsPerWeek (F=13,731) | AvgSessionDuration (F=5,084) | PlayerLevel (F=216) | Achievements (F=233)  
        **Dropped features:** Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty (F < 2, near-noise)
        """
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
