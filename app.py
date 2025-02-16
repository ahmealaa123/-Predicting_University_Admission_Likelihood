
import pandas as pd
import gradio as gr
import joblib

std=joblib.load('std_col.pkl')
lr=joblib.load('model.pkl')


std_col=['GPA', 'Test Score', 'Extracurricular Activities', 'Volunteer Hours','Recommendation Letters', 'Essay Score']




def predict_admission(gpa, ts, ea, vh, rl, es):
  try:
    input_data = pd.DataFrame(
        {
            "GPA":[gpa],
            "Test Score":[ts],
            "Extracurricular Activities":[ea],
            "Volunteer Hours":[vh],
            "Recommendation Letters":[rl],
            "Essay Score":[es]}
    )

    input_data[std_col]=std.transform(input_data[std_col])
    prediction = lr.predict(input_data)
    if prediction[0] == 0:
          return "No"
    else:
          return "Yes"
  except Exception as e:
        return str(e)
gr.Interface(
    inputs= [

             gr.Number(label="GPA"),
             gr.Number(label="Test Score"),
             gr.Number(label="Extracurricular Activities"),
             gr.Number(label="Volunteer Hours"),
             gr.Number(label="Recommendation Letters"),
             gr.Number(label="Essay Score")
    ],
    fn = predict_admission, outputs= gr.Textbox(label="Admission Decision")
).launch()