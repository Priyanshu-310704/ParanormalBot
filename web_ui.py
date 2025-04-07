import gradio as gr
from .analysis import ParanormalAnalyzer

# Initialize analyzer at module level
analyzer = ParanormalAnalyzer()

# Team information
TEAM_INFO = """
👥 Development Team:
1. Priyanshu (12303126) - Team Leader
2. Ankush (12300584)
3. Pintu (12313139)
"""

def analyze_text(text):
    try:
        result = analyzer.analyze(text)

        output = ["🔮 Paranormal Analysis Report", TEAM_INFO]  # Added team info here

        # Entity detection
        if result['entities']:
            output.append("\n📌 Detected Entities:")
            for ent in result['entities']:
                output.append(f"- {ent['text']} ({ent['type']})")
        else:
            output.append("\n🔍 No paranormal entities detected")

        # Similar cases with full text
        if result.get('similar_reports'):
            output.append("\n📚 Similar Historical Reports:")
            for i, report in enumerate(result['similar_reports'][:3], 1):
                output.append(f"\nCase #{i}:")
                output.append(f"Title: {report.get('title', 'Untitled')}")
                output.append(f"Location: {report.get('location', 'Unknown')}")
                output.append(f"Type: {report.get('type', 'Unspecified')}")
                output.append("\nFull Report:")
                output.append(report.get('text', 'No details available'))
                output.append(f"\nSimilarity Score: {report.get('similarity', 0):.2f}")
                output.append("-" * 50)

        return "\n".join(output)

    except Exception as e:
        return f"⚠️ Analysis error: {str(e)}\n{TEAM_INFO}"  # Added team info to error message too


# Create interface
iface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=10, placeholder="Describe your experience..."),
    outputs=gr.Textbox(label="Analysis Report"),
    title="👻 Paranormal Investigator Pro",
    description=f"A paranormal analysis tool developed by:\n{TEAM_INFO}",  # Added team info to description
    allow_flagging="never"
)


if __name__ == "__main__":
    iface.launch(server_port=7860, server_name="127.0.0.1")