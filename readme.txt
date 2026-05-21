在終端機執行：
cd F:\CodingTest\pic-enhance
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000


然後開啟瀏覽器訪問 http://127.0.0.1:8000


按 Ctrl+C 可停止伺服器


///

提醒一下：第一次上傳圖片處理時，模型權重會自動從 HuggingFace 下載（x2 約 64MB、x4 約 64MB），之後就會快取在 weights/ 目錄，不需要重複下載
