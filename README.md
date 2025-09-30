# Forecasting Fuel Efficiency: A Data-Driven Approach to Vehicle Emissions in Canada

This repository demonstrates a **machine learning-powered dashboard** that supports **consumers, policymakers, and manufacturers** in making informed decisions about **fuel efficiency, carbon emissions, and carbon tax impacts**. The project extends the **Canadian EnerGuide system** by integrating **predictive analytics, scalable data pipelines, and interactive dashboards**.

* **Dataset**: [Government of Canada – Fuel Consumption Ratings (2015–2025)](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
* **Live Demo**: [Forecasting Fuel Efficiency](https://forecasting-fuel-efficiency.onrender.com/)

---

## Key Features

* **Vehicle Search and Comparison** – Compare **fuel efficiency, emissions, and annual fuel costs** across models.
* **Interactive Dashboards** – Visualize **historical and forecasted trends** in fuel economy, emissions, and costs.
* **Predictive Analytics** – Machine learning models (e.g., **Scikit-learn, XGBoost**) estimate **fuel consumption and carbon tax impacts**.
* **Policy Insights** – Provide **projections for government agencies** to refine emissions policies.
* **Manufacturer Insights** – Identify **inefficient vehicle models** for compliance improvements.
* **Personalized Recommendations** – Suggest **cost-effective and eco-friendly vehicles** to consumers.

---

## Architecture

* **Data Ingestion** – Government datasets (CSV) + **synthetic streaming events** via **Kafka on Azure Event Hubs**.
* **Data Orchestration** – Automated ETL with **Apache Airflow DAGs** for ingestion, validation, and transformation.
* **Data Storage** – Raw and curated data stored in **Azure Data Lake Storage (ADLS Gen2)**.
* **Data Warehouse** – Processed datasets loaded into **Snowflake**, with **Snowpipe** for near real-time ingestion.
* **Visualization** – **Tableau** and **Streamlit dashboards** connected to Snowflake for interactive analytics.
* **Frontend** – User-facing dashboard built with **React/Vue.js**.

---

## Tech Stack

* **Backend**: Python, FastAPI
* **Machine Learning**: Scikit-learn, XGBoost, Random Forest, Decision Trees
* **Data Engineering**: Pandas, NumPy, **Apache Airflow**
* **Streaming**: **Kafka on Azure Event Hubs** (simulated ingestion)
* **Data Warehouse**: **Snowflake**
* **Visualization**: **Tableau, Streamlit, Matplotlib, Seaborn**
* **Frontend**: React or Vue.js
* **Deployment**: Render (demo hosting), **Azure (pipeline services)**

---

## Project Goals

* Improve **consumer awareness** of fuel costs and carbon tax.
* Enable **data-driven decision-making** for vehicle purchases.
* Provide **government agencies** with insights for emissions policies.
* Help **manufacturers** meet sustainability targets.

---

## Future Enhancements

* Integrate **real-time API/IoT data feeds**.
* Deploy predictive models as **APIs with Azure ML**.
* Expand coverage beyond Canada.
* Add **Power BI dashboards** for enterprise analytics.
