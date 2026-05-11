voicebot-ctm/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── mcp_server/
│   │   ├── server.py
│   │   │
│   │   └── tools/
│   │       ├── rag_tool.py
│   │       ├── reservation_tool.py
│   │       ├── consultation_tool.py
│   │       └── tracking_tool.py
│   │
│   ├── core/
│   │   ├── asr.py
│   │   ├── tts.py
│   │   ├── translator.py
│   │   ├── llm.py
│   │   └── memory.py
│   │
│   ├── rag/
│   │   ├── retrieval.py
│   │   ├── embeddings.py
│   │   └── faiss_store.py
│   │
│   ├── database/
│   │   ├── db.py
│   │   └── reservations.db
│   │
│   └── config/
│       └── settings.py
│
├── data/
│   │
│   ├── raw/
│   │   ├── voyages.txt
│   │   ├── colis.txt
│   │   └── politiques.txt
│   │
│   └── index/
│       ├── .gitkeep
│       └── README.md
│
├── experiments/
│   ├── build_index.ipynb
│   ├── tts_training.ipynb
│   └── tests_colab.ipynb
│
├── models/
│   ├── asr/
│   ├── tts/
│   └── embedding/
│
├── logs/
│   └── conversations.log
│
└── tests/
    ├── test_rag.py
    ├── test_database.py
    └── test_asr.py