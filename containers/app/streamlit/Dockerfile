FROM tensorflow/tensorflow

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

CMD streamlit run --server.port 4000 app.py
