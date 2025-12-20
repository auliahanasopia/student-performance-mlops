Workflow Eksperimen SMSML 

1. Dataset mentah disimpan pada folder namadataset_raw.
2. Proses preprocessing dilakukan menggunakan script automate_Aulia-Hana-Sophiah.py.
3. Dataset hasil preprocessing disimpan pada folder namadataset_preprocessing.
4. Model dilatih dan dituning menggunakan modelling_tuning.py.
5. Seluruh eksperimen dicatat menggunakan MLflow dan DagsHub.
6. Pipeline CI dijalankan menggunakan GitHub Actions untuk memastikan preprocessing dan training berjalan otomatis.
