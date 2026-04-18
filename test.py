import base64
import pandas as pd
from rail_madad_model import RailMadadModel

model = RailMadadModel()
model.load_context(None)  # initializes self.engine

with open("ticket.webp", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

df = pd.DataFrame([{
    "query": "",
    "train_number": "",
    "pnr": "",
    "issue_type": "",
    "language_code": "",
    "ticket_image_base64": image_base64,
    "user_id": "user123",
}])

print(model.predict(context=None, model_input=df).to_dict(orient="records"))
