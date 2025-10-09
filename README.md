# Real-Time ML Pipeline for Fuel Efficiency Forecasting

This project demonstrates a complete, end-to-end streaming machine learning pipeline for predicting vehicle fuel efficiency. It uses Kafka (via Redpanda) to simulate a real-time data stream, processes the data through a Bronze/Silver architecture, and continuously retrains a model. A Streamlit dashboard provides a live view of the model's performance.

<!-- TODO: Add a screenshot of your dashboard here! -->

## Features

- **Real-Time Data Ingestion**: A Python script (`producer.py`) reads vehicle data and streams it into Kafka.
- **Kafka Message Bus**: Redpanda is used as a lightweight, Docker-based Kafka-compatible message broker.
- **Bronze/Silver Data Layers**:
  - **Bronze**: Raw, unfiltered data is consumed from Kafka and stored in Parquet files.
  - **Silver**: Data is cleaned, transformed, and enriched, creating an analysis-ready dataset.
- **Continuous Model Training**: A trainer script (`consumer_trainer.py`) monitors the Silver layer and retrains a Scikit-learn model whenever new data arrives.
- **Live Monitoring Dashboard**: A Streamlit application (`training_monitor.py`) visualizes the model's performance (R² and MAE) in real-time and allows for pipeline control.

## Project Structure

```
├── dashboard/
│   └── training_monitor.py   # Streamlit monitoring UI
├── data/
│   └── Cars_Cleaned.xlsx     # Source data
├── ingestion/
│   └── producer.py           # Streams data into Kafka
├── transform/
│   ├── consumer_to_parquet.py # Bronze layer writer
│   ├── consumer_to_silver.py  # Silver layer writer
│   └── consumer_trainer.py    # Model trainer
├── docker-compose.kafka.yml  # Docker Compose for Redpanda (Kafka)
├── requirements.txt          # Python dependencies
├── .gitignore                # Files to ignore for Git
├── .env.example              # Environment variable template
└── README.md                 # This file
```

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- An environment manager (like `venv` or `conda`)


## How to Run

1.  **Clone the repository:** (Replace `<your-github-username>` with your actual username)
    ```bash
    git clone https://github.com/<your-github-username>/streaming-fuel-forecasting.git
    cd streaming-fuel-forecasting
    ```

2.  **Set up a Python environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure Environment (Optional):**
    The project is configured to run locally out-of-the-box. If you need to change the Kafka topic or server address, you can copy the `.env.example` file to a new file named `.env` and modify the values.
    ```bash
    # On Windows (Command Prompt)
    copy .env.example .env
    ```
3.  **Launch the Streamlit Dashboard:**
    The easiest way to run the entire pipeline is from the dashboard's UI.
    ```bash
    streamlit run dashboard/training_monitor.py
    ```

4.  **Start the Pipeline:**
    - Open the Streamlit app in your browser (usually `http://localhost:8501`).
    - Click the **"🚀 Start Pipeline"** button. This will:
      - Start the Kafka service using Docker.
      - Launch the producer, consumers (Bronze/Silver), and the model trainer as background processes.
    - You will see the performance charts and KPIs update in real-time as data flows through the system.

5.  **Stop the Pipeline:**
    - When you are finished, click the **"🛑 Stop Pipeline"** button in the dashboard to shut down all processes and the Kafka container gracefully.

### Alternative: Running with PowerShell (Windows)

You can also run all components using the provided PowerShell script. Right-click `run_pipeline.ps1` and select "Run with PowerShell". Note that this does not provide the graceful shutdown offered by the Streamlit UI.

## Deploying to Render

This project can be deployed to Render using the provided `render.yaml` configuration file.

1.  **Push to GitHub:** Make sure all your latest changes, including `render.yaml`, are pushed to your GitHub repository.

2.  **Create a New Blueprint on Render:**
    - In your Render dashboard, click **New +** and select **Blueprint**.
    - Connect the GitHub repository containing this project.
    - Render will automatically detect and parse the `render.yaml` file.

3.  **Apply and Deploy:**
    - Review the services that Render will create (a web service, a private service for Kafka, and four background workers).
    - Click **Apply** to start the deployment. Render will build and launch all the services.

**Note on Render's Free Tier:**
- The free tier for web services and background workers may "spin down" due to inactivity, which will interrupt the pipeline.
- The free tier for private services is very limited. Running Kafka/Redpanda may require a paid plan for stability.
- The file system is ephemeral, meaning all data in the `data/` directory (Bronze, Silver, metrics) will be lost on restarts and redeploys. For a production setup, you would need to switch to a persistent storage solution like Amazon S3.