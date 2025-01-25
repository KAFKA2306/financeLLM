
(signate) M:\ML\signatejpx>python py/ragout.py
2025-01-25 15:40:50,801 - INFO - Directory structure verified and created
2025-01-25 15:40:50,801 - INFO - Initializing EnhancedRAG system...
2025-01-25 15:40:55,344 - INFO - Load pretrained SentenceTransformer: intfloat/multilingual-e5-large
2025-01-25 15:40:59,132 - INFO - Loading configuration from: M:\ML\signatejpx\secret\config.py
2025-01-25 15:40:59,133 - INFO - Initialization completed successfully
2025-01-25 15:40:59,136 - INFO - Loaded 100 queries
2025-01-25 15:40:59,136 - INFO - Loading documents from: M:\ML\signatejpx\data\documents\documents
Loading PDFs:   0%|                      | 0/19 [00:00<?, ?it/s]2025-01-25 15:41:00,557 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\1.pdf (14 pages) in 1.42 seconds  
Loading PDFs:   5%|▋             | 1/19 [00:01<00:25,  1.42s/it]2025-01-25 15:41:13,284 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\10.pdf (160 pages) in 12.72 seconds
Loading PDFs:  11%|█▍            | 2/19 [00:14<02:17,  8.07s/it]2025-01-25 15:41:19,369 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\11.pdf (40 pages) in 6.08 seconds 
Loading PDFs:  16%|██▏           | 3/19 [00:20<01:54,  7.16s/it]2025-01-25 15:41:27,602 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\12.pdf (36 pages) in 8.23 seconds 
Loading PDFs:  21%|██▉           | 4/19 [00:28<01:53,  7.59s/it]2025-01-25 15:41:33,311 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\13.pdf (70 pages) in 5.71 seconds 
Loading PDFs:  26%|███▋          | 5/19 [00:34<01:36,  6.91s/it]2025-01-25 15:41:33,321 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\14.pdf: PyCryptodome is required for AES algorithm
2025-01-25 15:41:33,332 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\15.pdf: PyCryptodome is required for AES algorithm
2025-01-25 15:41:52,825 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\16.pdf (109 pages) in 19.49 seconds
Loading PDFs:  42%|█████▉        | 8/19 [00:53<01:13,  6.66s/it]2025-01-25 15:41:52,844 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\17.pdf: PyCryptodome is required for AES algorithm    
2025-01-25 15:41:59,312 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\18.pdf (70 pages) in 6.47 secondsLoading PDFs:  53%|██████▊      | 10/19 [01:00<00:48,  5.41s/it]2025-01-25 15:41:59,343 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\19.pdf: PyCryptodome is required for AES algorithm
2025-01-25 15:41:59,355 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\2.pdf: PyCryptodome is required for AES algorithm
2025-01-25 15:42:07,791 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\3.pdf (164 pages) in 8.44 secondsLoading PDFs:  68%|████████▉    | 13/19 [01:08<00:25,  4.27s/it]2025-01-25 15:42:13,418 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\4.pdf (46 pages) in 5.62 seconds
Loading PDFs:  74%|█████████▌   | 14/19 [01:14<00:22,  4.51s/it]2025-01-25 15:42:13,425 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\5.pdf: PyCryptodome is required for AES algorithm     
2025-01-25 15:42:13,432 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\6.pdf: PyCryptodome is required for AES algorithm
2025-01-25 15:42:17,966 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\7.pdf (90 pages) in 4.53 seconds
Loading PDFs:  89%|███████████▋ | 17/19 [01:18<00:06,  3.23s/it]C:\Users\100ca\anaconda3\envs\signate\Lib\site-packages\PyPDF2\_cmap.py:142: PdfReadWarning: Advanced encoding /UniJIS-UTF16-H not implemented yet  
  warnings.warn(
2025-01-25 15:42:21,244 - INFO - Successfully read PDF M:\ML\signatejpx\data\documents\documents\8.pdf (237 pages) in 3.28 seconds
Loading PDFs:  95%|████████████▎| 18/19 [01:22<00:03,  3.24s/it]2025-01-25 15:42:21,265 - ERROR - Error reading PDF M:\ML\signatejpx\data\documents\documents\9.pdf: PyCryptodome is required for AES algorithm     
Loading PDFs: 100%|█████████████| 19/19 [01:22<00:00,  4.32s/it]
2025-01-25 15:42:21,265 - INFO - Document loading completed: 11 successful, 8 failed
2025-01-25 15:42:21,266 - INFO - Loaded 11 documents 
2025-01-25 15:42:21,266 - INFO - Starting query processing for 100 queries
2025-01-25 15:42:21,266 - INFO - Creating vector store...
2025-01-25 16:25:51,847 - INFO - Loading faiss with AVX2 support.
2025-01-25 16:25:51,892 - INFO - Successfully loaded faiss with AVX2 support.
2025-01-25 16:25:52,407 - INFO - Vector store created with 2084 chunks
Processing queries:   0%|   | 0/100 [00:00<?, ?it/s]2025-01-25 16:25:52,410 - INFO - Processing query: 高 松コンストラクショングループの2025年3月期の受注高の計画は前期比何倍か、小数第三位を四捨五入し答えてくださ い。
2025-01-25 16:26:00,936 - INFO - API request completed in 8.18 seconds
2025-01-25 16:26:00,940 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162600.json
Processing queries:   1%| | 1/100 [00:08<14:04,  8.52025-01-25 16:26:00,940 - INFO - Processing query: 株 式会社キッツの取締役の報酬のうち株式報酬の割合は何％ ？
2025-01-25 16:26:20,077 - INFO - API request completed in 18.66 seconds
2025-01-25 16:26:20,081 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162620.json
Processing queries:   2%| | 2/100 [00:27<24:07, 14.72025-01-25 16:26:20,083 - INFO - Processing query: グ ローリーが2024年の統合報告書の中で、研究開発費が増加 した理由として挙げている事由は何？
2025-01-25 16:26:47,172 - INFO - API request completed in 26.80 seconds
2025-01-25 16:26:47,176 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162647.json
Processing queries:   3%| | 3/100 [00:54<32:58, 20.42025-01-25 16:26:47,177 - INFO - Processing query: 高 松コンストラクショングループの2024年3月期の非常勤監査役は何人？
2025-01-25 16:27:03,760 - INFO - API request completed in 16.14 seconds
2025-01-25 16:27:03,764 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162703.json
Processing queries:   4%| | 4/100 [01:11<30:13, 18.82025-01-25 16:27:03,765 - INFO - Processing query: 2023年で即席めんの一人当たりの年間消費量が最も多い国はどこか。
2025-01-25 16:27:14,224 - INFO - API request completed in 10.15 seconds
2025-01-25 16:27:14,229 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162714.json
Processing queries:   5%| | 5/100 [01:21<25:06, 15.82025-01-25 16:27:14,230 - INFO - Processing query: 株 式会社キッツで2023年度のデジタル教育受講者数は何人？ 
2025-01-25 16:27:22,447 - INFO - API request completed in 7.75 seconds
2025-01-25 16:27:22,452 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162722.json
Processing queries:   6%| | 6/100 [01:30<20:46, 13.22025-01-25 16:27:22,453 - INFO - Processing query: 2023年度の日清食品ホールディングスの海外事業において、コア営業利益が2番目に高い地域に含まれる国として記載がある国名を全て教えてください。
2025-01-25 16:27:39,318 - INFO - API request completed in 16.51 seconds
2025-01-25 16:27:39,328 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162739.json
Processing queries:   7%| | 7/100 [01:46<22:23, 14.42025-01-25 16:27:39,330 - INFO - Processing query: 株 式会社キッツの2023年度売上高構成比が2番目に高い事業の製造・販売会社を全て上げよ。（回答には株式会社も含め ること）
2025-01-25 16:27:55,684 - INFO - API request completed in 15.92 seconds
2025-01-25 16:27:55,689 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162755.json
Processing queries:   8%| | 8/100 [02:03<23:04, 15.02025-01-25 16:27:55,690 - INFO - Processing query: IHIの2023年度の資源・エネルギー・環境事業領域の営業利益 が過去最高を記録した要因として挙げているものはなにか 
2025-01-25 16:28:26,445 - ERROR - API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)       
2025-01-25 16:28:26,446 - ERROR - Error processing query 8: API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)
Processing queries:   9%| | 9/100 [02:34<30:16, 19.92025-01-25 16:28:26,448 - INFO - Processing query: IHIグループの経営目標の進捗の2022年度と2023年度を比較し たとき、営業利益率と税引後ROICではどちらの方が差が大 きいか
2025-01-25 16:28:49,113 - INFO - API request completed in 22.21 seconds
2025-01-25 16:28:49,118 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162849.json
Processing queries:  10%| | 10/100 [02:56<31:11, 20.2025-01-25 16:28:49,119 - INFO - Processing query: IHIが燃料アンモニアバリューチェーン構築を支える強みとし て挙げていることは何ですか。
2025-01-25 16:29:10,448 - INFO - API request completed in 20.74 seconds
2025-01-25 16:29:10,453 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162910.json
Processing queries:  11%| | 11/100 [03:18<31:05, 20.2025-01-25 16:29:10,454 - INFO - Processing query: 2023年の即席めんの一人当たりの年間消費量が最も大きい国と日本の差分は何食か。
2025-01-25 16:29:34,454 - INFO - API request completed in 23.78 seconds
2025-01-25 16:29:34,459 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162934.json
Processing queries:  12%| | 12/100 [03:42<32:06, 21.2025-01-25 16:29:34,460 - INFO - Processing query: メ ディアドゥの2024年2月期のエンジニアの人数は？        
2025-01-25 16:29:41,837 - INFO - API request completed in 7.19 seconds
2025-01-25 16:29:41,841 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162941.json
Processing queries:  13%|▏| 13/100 [03:49<25:21, 17.2025-01-25 16:29:41,842 - INFO - Processing query: 全 国保証株式会社の2024/3期の新規採用数は何人？
2025-01-25 16:29:47,100 - INFO - API request completed in 4.93 seconds
2025-01-25 16:29:47,105 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162947.json
Processing queries:  14%|▏| 14/100 [03:54<19:46, 13.2025-01-25 16:29:47,106 - INFO - Processing query: ラ イフコーポレーションのROEは2022年度から2023年度にかけて向上したのは何％か。
2025-01-25 16:29:57,936 - INFO - API request completed in 10.61 seconds
2025-01-25 16:29:57,941 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_162957.json
Processing queries:  15%|▏| 15/100 [04:05<18:17, 12.2025-01-25 16:29:57,942 - INFO - Processing query: 2024年3月期のグローリーのセグメント別売上高の中で、2番目に売上高が大きいセグメントの販売先を答えよ。
2025-01-25 16:30:15,432 - INFO - API request completed in 17.18 seconds
2025-01-25 16:30:15,435 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163015.json
Processing queries:  16%|▏| 16/100 [04:23<20:00, 14.2025-01-25 16:30:15,436 - INFO - Processing query: カ ゴメ単体における2023年度の新入社員採用数は2022年度と 比べると増加したか、減少したか？
2025-01-25 16:30:24,712 - INFO - API request completed in 9.04 seconds
2025-01-25 16:30:24,717 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163024.json
Processing queries:  17%|▏| 17/100 [04:32<17:40, 12.2025-01-25 16:30:24,718 - INFO - Processing query: 日 産自動車の2023年度の従業員の平均年収は約何万円でしょ うか。
2025-01-25 16:30:54,144 - INFO - API request completed in 29.18 seconds
2025-01-25 16:30:54,148 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163054.json
Processing queries:  18%|▏| 18/100 [05:01<24:18, 17.2025-01-25 16:30:54,149 - INFO - Processing query: 高 松コンストラクショングループの2024年3月31日時点での発行済株式の総数は何株ですか
2025-01-25 16:31:09,178 - INFO - API request completed in 14.44 seconds
2025-01-25 16:31:09,181 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163109.json
Processing queries:  19%|▏| 19/100 [05:16<22:53, 16.2025-01-25 16:31:09,182 - INFO - Processing query: ハ ウス食品グループの2024年3月期の売上が2番目に大きいセ グメントはなにか。
2025-01-25 16:31:28,355 - INFO - API request completed in 18.82 seconds
2025-01-25 16:31:28,360 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163128.json
Processing queries:  20%|▏| 20/100 [05:35<23:30, 17.2025-01-25 16:31:28,361 - INFO - Processing query: 日 清食品ホールディングスが所有する国内外のグループ生産 拠点の全部で何拠点か。
2025-01-25 16:31:34,778 - INFO - API request completed in 6.17 seconds
2025-01-25 16:31:34,782 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163134.json
Processing queries:  21%|▏| 21/100 [05:42<18:46, 14.2025-01-25 16:31:34,783 - INFO - Processing query: ラ イフコーポレーションが提供するライフアプリの2023年度 末の会員数は約何万人？
2025-01-25 16:31:43,327 - INFO - API request completed in 8.04 seconds
2025-01-25 16:31:43,331 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163143.json
Processing queries:  22%|▏| 22/100 [05:50<16:18, 12.2025-01-25 16:31:43,332 - INFO - Processing query: パ ナソニックグループにおいて、2023年度企業市民活動の費 用が高いのは、北米と社会福祉分野のどちらか。
2025-01-25 16:31:47,097 - INFO - API request completed in 3.44 seconds
2025-01-25 16:31:47,100 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163147.json
Processing queries:  23%|▏| 23/100 [05:54<12:43,  9.2025-01-25 16:31:47,101 - INFO - Processing query: 日 産自動車のコーポレートパーパスは何ですか。
2025-01-25 16:31:52,162 - INFO - API request completed in 4.84 seconds
2025-01-25 16:31:52,167 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163152.json
Processing queries:  24%|▏| 24/100 [05:59<10:42,  8.2025-01-25 16:31:52,168 - INFO - Processing query: ラ イフコーポレーションの2023年度の店舗数は何店舗か。   
2025-01-25 16:31:56,800 - INFO - API request completed in 3.96 seconds
2025-01-25 16:31:56,805 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163156.json
Processing queries:  25%|▎| 25/100 [06:04<09:08,  7.2025-01-25 16:31:56,806 - INFO - Processing query: 東 急不動産の2023年度の営業利益において、資産活用型ビジ ネスのうち戦略投資事業が占める割合は何％か、小数第二 位を四捨五入して答えてください。
2025-01-25 16:32:20,322 - INFO - API request completed in 22.88 seconds
2025-01-25 16:32:20,327 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163220.json
Processing queries:  26%|▎| 26/100 [06:27<15:01, 12.2025-01-25 16:32:20,328 - INFO - Processing query: 4℃ ホールディングスビジネススクールは最短で卒業した卒業 生は何年で卒業した？
2025-01-25 16:32:39,358 - INFO - API request completed in 15.38 seconds
2025-01-25 16:32:39,366 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163239.json
Processing queries:  27%|▎| 27/100 [06:46<17:19, 14.2025-01-25 16:32:39,367 - INFO - Processing query: カ ゴメが人材育成において重要視している観点を全てあげて ください。
2025-01-25 16:33:12,299 - ERROR - API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)       
2025-01-25 16:33:12,300 - ERROR - Error processing query 27: API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)
Processing queries:  28%|▎| 28/100 [07:19<23:48, 19.2025-01-25 16:33:12,301 - INFO - Processing query: パ ナソニックグループのCO2ゼロ工場数において、2021年度までに何工場実現しましたか。
2025-01-25 16:33:19,738 - INFO - API request completed in 4.94 seconds
2025-01-25 16:33:19,743 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163319.json
Processing queries:  29%|▎| 29/100 [07:27<19:04, 16.2025-01-25 16:33:19,745 - INFO - Processing query: モ スグループの2024年3月31日現在の海外店舗数は何店舗？  
2025-01-25 16:33:43,645 - INFO - API request completed in 19.72 seconds
2025-01-25 16:33:43,648 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163343.json
Processing queries:  30%|▎| 30/100 [07:51<21:32, 18.2025-01-25 16:33:43,649 - INFO - Processing query: ハ ウス食品グループの従業員数は、国内と海外、どちらが多 いですか？
2025-01-25 16:34:02,759 - INFO - API request completed in 17.34 seconds
2025-01-25 16:34:02,764 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163402.json
Processing queries:  31%|▎| 31/100 [08:10<21:27, 18.2025-01-25 16:34:02,765 - INFO - Processing query: 全 国保証株式会社の平均勤続年数は2024/3期は2023/3期対比 何年伸びましたか？
2025-01-25 16:34:25,444 - INFO - API request completed in 20.04 seconds
2025-01-25 16:34:25,450 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163425.json
Processing queries:  32%|▎| 32/100 [08:33<22:30, 19.2025-01-25 16:34:25,451 - INFO - Processing query: ハ ウス食品グループの系列VCをすべて挙げてください。     
2025-01-25 16:34:30,940 - INFO - API request completed in 3.57 seconds
2025-01-25 16:34:30,944 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163430.json
Processing queries:  33%|▎| 33/100 [08:38<17:22, 15.2025-01-25 16:34:30,945 - INFO - Processing query: 明 治グループの2023年度のROICが一番大きいセグメントは全 体のROICと比較して何％大きいですか。
2025-01-25 16:34:51,949 - INFO - API request completed in 19.27 seconds
2025-01-25 16:34:51,954 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163451.json
Processing queries:  34%|▎| 34/100 [08:59<18:54, 17.2025-01-25 16:34:51,956 - INFO - Processing query: サ ントリーグループサステナビリティサイトにおいて、KPMG あずさサステナビリティ株式会社による第三者保証の対象 となっている数値はいくつありますか。
2025-01-25 16:35:24,709 - ERROR - API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)       
2025-01-25 16:35:24,710 - ERROR - Error processing query 34: API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)
Processing queries:  35%|▎| 35/100 [09:32<23:40, 21.2025-01-25 16:35:24,710 - INFO - Processing query: カ ゴメの加工用トマトが潰れにくい理由となっている特徴を2つ答えてください。
2025-01-25 16:35:42,524 - INFO - API request completed in 16.09 seconds
2025-01-25 16:35:42,529 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163542.json
Processing queries:  36%|▎| 36/100 [09:50<22:01, 20.2025-01-25 16:35:42,530 - INFO - Processing query: 東 急不動産のITパスポート取得率は2023年度は2022年度対比 何％増加した？
2025-01-25 16:35:57,738 - INFO - API request completed in 11.70 seconds
2025-01-25 16:35:57,744 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163557.json
Processing queries:  37%|▎| 37/100 [10:05<19:58, 19.2025-01-25 16:35:57,745 - INFO - Processing query: モ スグループが紙製パッケージに使用している紙は何認証紙 か。
2025-01-25 16:36:08,931 - INFO - API request completed in 9.22 seconds
2025-01-25 16:36:08,935 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163608.json
Processing queries:  38%|▍| 38/100 [10:16<17:13, 16.2025-01-25 16:36:08,936 - INFO - Processing query: 東 急不動産のDX推進人財のうち、中級に位置付けられる人財 の呼称は何？
2025-01-25 16:36:26,988 - INFO - API request completed in 16.06 seconds
2025-01-25 16:36:26,994 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163626.json
Processing queries:  39%|▍| 39/100 [10:34<17:22, 17.2025-01-25 16:36:26,996 - INFO - Processing query: IHIが統合報告書2024において成長事業と位置付けた事業で、 開発を行っている国として挙げられている国名を全て答え よ。
2025-01-25 16:36:40,063 - INFO - API request completed in 10.57 seconds
2025-01-25 16:36:40,068 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163640.json
Processing queries:  40%|▍| 40/100 [10:47<15:52, 15.2025-01-25 16:36:40,069 - INFO - Processing query: ハ ウス食品グループの2024年3月期の営業利益率は2023年3月 比何％上昇したか
2025-01-25 16:37:04,355 - INFO - API request completed in 22.62 seconds
2025-01-25 16:37:04,359 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163704.json
Processing queries:  41%|▍| 41/100 [11:11<18:05, 18.2025-01-25 16:37:04,360 - INFO - Processing query: サ ントリー美術館とサントリーホールでは、どちらの開館が 先ですか？
2025-01-25 16:37:13,030 - INFO - API request completed in 7.11 seconds
2025-01-25 16:37:13,035 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163713.json
Processing queries:  42%|▍| 42/100 [11:20<14:58, 15.2025-01-25 16:37:13,036 - INFO - Processing query: 日 産自動車の2023年度の指定化学物質数は2019年度と比較し ていくつ増加したか。
2025-01-25 16:37:36,436 - INFO - API request completed in 19.57 seconds
2025-01-25 16:37:36,440 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163736.json
Processing queries:  43%|▍| 43/100 [11:44<16:58, 17.2025-01-25 16:37:36,441 - INFO - Processing query: ク レハのサステナビリティ委員会は原則として年何回開催さ れますか？
2025-01-25 16:37:48,728 - INFO - API request completed in 10.90 seconds
2025-01-25 16:37:48,732 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163748.json
Processing queries:  44%|▍| 44/100 [11:56<15:06, 16.2025-01-25 16:37:48,733 - INFO - Processing query: 東 急不動産の社内ベンチャー制度の名前は？
2025-01-25 16:38:08,442 - INFO - API request completed in 16.95 seconds
2025-01-25 16:38:08,447 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163808.json
Processing queries:  45%|▍| 45/100 [12:16<15:48, 17.2025-01-25 16:38:08,448 - INFO - Processing query: モ スグループの2023年度の店舗数が最も多い海外拠点はどこ か
2025-01-25 16:38:22,227 - INFO - API request completed in 11.54 seconds
2025-01-25 16:38:22,233 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163822.json
Processing queries:  46%|▍| 46/100 [12:29<14:35, 16.2025-01-25 16:38:22,234 - INFO - Processing query: 東 急不動産のセグメントのうち、最も2023年度の総資産が多 いセグメントが提供している価値を全てあげてください。 
2025-01-25 16:38:54,109 - ERROR - API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)       
2025-01-25 16:38:54,110 - ERROR - Error processing query 46: API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)
Processing queries:  47%|▍| 47/100 [13:01<18:28, 20.2025-01-25 16:38:54,111 - INFO - Processing query: モ スグループの2023年度の連結売上高は2022年度対比何％向 上した？小数第二位で四捨五入してください。
2025-01-25 16:39:13,363 - INFO - API request completed in 13.92 seconds
2025-01-25 16:39:13,368 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163913.json
Processing queries:  48%|▍| 48/100 [13:20<17:41, 20.2025-01-25 16:39:13,369 - INFO - Processing query: メ ディアドゥは電子書籍市場を「何期」に移行しつつあると 考えていますか？
2025-01-25 16:39:43,898 - INFO - API request completed in 28.24 seconds
2025-01-25 16:39:43,903 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163943.json
Processing queries:  49%|▍| 49/100 [13:51<19:55, 23.2025-01-25 16:39:43,904 - INFO - Processing query: メ ディアドゥが取引する出版社は何社以上？
2025-01-25 16:39:59,866 - INFO - API request completed in 14.62 seconds
2025-01-25 16:39:59,871 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_163959.json
Processing queries:  50%|▌| 50/100 [14:07<17:40, 21.2025-01-25 16:39:59,872 - INFO - Processing query: 全 国保証株式会社の有給休暇取得日数は2023/3期対比2024/3 期は何日増加しましたか？
2025-01-25 16:40:23,257 - INFO - API request completed in 21.45 seconds
2025-01-25 16:40:23,261 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164023.json
Processing queries:  51%|▌| 51/100 [14:30<17:51, 21.2025-01-25 16:40:23,262 - INFO - Processing query: グ ローリーは2020中計を「種まき」、2023中計を「育成」時 期と位置付けているが、2026中計の位置づけは何？       
2025-01-25 16:40:43,915 - INFO - API request completed in 17.61 seconds
2025-01-25 16:40:43,922 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164043.json
Processing queries:  52%|▌| 52/100 [14:51<17:12, 21.2025-01-25 16:40:43,923 - INFO - Processing query: 全 国保証株式会社が民間金融機関の住宅ローン保証業務を開 始した年に挙げられている社会・経済の動きを全て答えよ 。
2025-01-25 16:40:54,267 - INFO - API request completed in 8.10 seconds
2025-01-25 16:40:54,272 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164054.json
Processing queries:  53%|▌| 53/100 [15:01<14:13, 18.2025-01-25 16:40:54,274 - INFO - Processing query: グ ローリーは国内外のサービス拠点並びに現地法人に約何名 のテクニカルスタッフを配置している？
2025-01-25 16:41:06,330 - INFO - API request completed in 10.33 seconds
2025-01-25 16:41:06,336 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164106.json
Processing queries:  54%|▌| 54/100 [15:13<12:31, 16.2025-01-25 16:41:06,337 - INFO - Processing query: 東 洋エンジニアリングの代表取締役が述べる、「当社の事業 に内在しているリスク」として具体的に述べられているも のを全てあげてください。
2025-01-25 16:41:28,626 - INFO - API request completed in 18.78 seconds
2025-01-25 16:41:28,633 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164128.json
Processing queries:  55%|▌| 55/100 [15:36<13:35, 18.2025-01-25 16:41:28,635 - INFO - Processing query: meijiサステナブルプロダクツ認定制度に基づきサステナブル プロダクツに認定されるには5つの評価基準のうちいくつクリアする必要がありますか？
2025-01-25 16:41:57,494 - INFO - API request completed in 25.86 seconds
2025-01-25 16:41:57,501 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164157.json
Processing queries:  56%|▌| 56/100 [16:05<15:39, 21.2025-01-25 16:41:57,502 - INFO - Processing query: IHIの2023年度における株主・投資家との主な対話テーマには どの様な分類があるか。
2025-01-25 16:42:29,745 - ERROR - API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)       
2025-01-25 16:42:29,747 - ERROR - Error processing query 56: API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)
Processing queries:  57%|▌| 57/100 [16:37<17:38, 24.2025-01-25 16:42:29,748 - INFO - Processing query: グ ローリーが、複数年度に渡り業績にプラスの影響があった として、挙げている事象を全て答えよ。
2025-01-25 16:42:55,437 - INFO - API request completed in 24.43 seconds
2025-01-25 16:42:55,442 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164255.json
Processing queries:  58%|▌| 58/100 [17:03<17:27, 24.2025-01-25 16:42:55,442 - INFO - Processing query: ラ イフコーポレーションの経営理念は何ですか。
2025-01-25 16:43:28,338 - ERROR - API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)       
2025-01-25 16:43:28,339 - ERROR - Error processing query 58: API request failed: HTTPSConnectionPool(host='hitachifibu.highreso.jp', port=10444): Read timed out. (read timeout=30)
Processing queries:  59%|▌| 59/100 [17:35<18:40, 27.2025-01-25 16:43:28,340 - INFO - Processing query: 日 産自動車の2023年度において、グローバル生産台数は何台 か。
2025-01-25 16:43:59,783 - INFO - API request completed in 29.77 seconds
2025-01-25 16:43:59,788 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164359.json
Processing queries:  60%|▌| 60/100 [18:07<19:02, 28.2025-01-25 16:43:59,789 - INFO - Processing query: 高 松コンストラクショングループの新卒採用者数実績値で2021/3期以降最も多い年の採用数は何人か
2025-01-25 16:44:17,544 - INFO - API request completed in 16.29 seconds
2025-01-25 16:44:17,550 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164417.json
Processing queries:  61%|▌| 61/100 [18:25<16:27, 25.2025-01-25 16:44:17,551 - INFO - Processing query: メ ディアドゥのサスティナビリティ推進事務局はどこの部署 が担当していますか？
2025-01-25 16:44:28,165 - INFO - API request completed in 8.21 seconds
2025-01-25 16:44:28,170 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164428.json
Processing queries:  62%|▌| 62/100 [18:35<13:14, 20.2025-01-25 16:44:28,171 - INFO - Processing query: 東 洋エンジニアリングに2023年度リファラル採用で入社した 社員は何名？
2025-01-25 16:44:44,327 - INFO - API request completed in 13.79 seconds
2025-01-25 16:44:44,332 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164444.json
Processing queries:  63%|▋| 63/100 [18:51<12:00, 19.2025-01-25 16:44:44,333 - INFO - Processing query: パ ナソニックグループの2023年度の事業活動におけるCO2排出量が最も多い海外地域はどこ？
2025-01-25 16:45:02,938 - INFO - API request completed in 16.88 seconds
2025-01-25 16:45:02,942 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164502.json
Processing queries:  64%|▋| 64/100 [19:10<11:32, 19.2025-01-25 16:45:02,943 - INFO - Processing query: 4℃ ホールディングスのアパレル事業「パレット」の2024年度 と2023年度を比較して、増加した店舗数は何店舗か？     
2025-01-25 16:45:17,042 - INFO - API request completed in 11.69 seconds
2025-01-25 16:45:17,051 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164517.json
Processing queries:  65%|▋| 65/100 [19:24<10:19, 17.2025-01-25 16:45:17,052 - INFO - Processing query: ク レハの年次有給休暇取得率は2022年から2023年で何％増加 した？
2025-01-25 16:45:38,020 - INFO - API request completed in 19.42 seconds
2025-01-25 16:45:38,026 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164538.json
Processing queries:  66%|▋| 66/100 [19:45<10:34, 18.2025-01-25 16:45:38,027 - INFO - Processing query: 明 治ホールディングスの海外の売上高において、2013年度か ら2023年度までの11年間で、食品セグメントが医薬品セグ メントを下回った年度を全てあげてください。
2025-01-25 16:45:57,040 - INFO - API request completed in 16.64 seconds
2025-01-25 16:45:57,046 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164557.json
Processing queries:  67%|▋| 67/100 [20:04<10:19, 18.2025-01-25 16:45:57,047 - INFO - Processing query: ク レハのエネルギー起源CO2排出量について、グループ会社を含めない場合、2013年度の排出量と比較して2023年度の排 出量は何％か、小数点第1位までの数字で四捨五入して答えてください。
2025-01-25 16:46:24,929 - INFO - API request completed in 24.99 seconds
2025-01-25 16:46:24,934 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164624.json
Processing queries:  68%|▋| 68/100 [20:32<11:28, 21.2025-01-25 16:46:24,935 - INFO - Processing query: サ ントリーグループの2023年の新規入社者数のうち、新卒の 占める割合は何％か、小数第二位で四捨五入して答えてく ださい。
2025-01-25 16:46:34,661 - INFO - API request completed in 7.12 seconds
2025-01-25 16:46:34,666 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164634.json
Processing queries:  69%|▋| 69/100 [20:42<09:17, 17.2025-01-25 16:46:34,667 - INFO - Processing query: サ ントリーグループが考えるサステナビリティ経営のテーマ をすべて挙げてください。
2025-01-25 16:46:53,760 - INFO - API request completed in 17.39 seconds
2025-01-25 16:46:53,765 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164653.json
Processing queries:  70%|▋| 70/100 [21:01<09:09, 18.2025-01-25 16:46:53,766 - INFO - Processing query: ク レハの特例子会社の名前は何か、株式会社を含む正式名称 で答えてください。
2025-01-25 16:47:04,338 - INFO - API request completed in 8.51 seconds
2025-01-25 16:47:04,343 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164704.json
Processing queries:  71%|▋| 71/100 [21:11<07:43, 15.2025-01-25 16:47:04,344 - INFO - Processing query: 健 康パナソニックの推進責任者の正式名称をカタカナで答え てください。
2025-01-25 16:47:13,122 - INFO - API request completed in 7.25 seconds
2025-01-25 16:47:13,127 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164713.json
Processing queries:  72%|▋| 72/100 [21:20<06:27, 13.2025-01-25 16:47:13,128 - INFO - Processing query: ク レハグループでは「エネルギー消費原単位」をどのように 算出していますか。
2025-01-25 16:47:42,113 - INFO - API request completed in 27.24 seconds
2025-01-25 16:47:42,117 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164742.json
Processing queries:  73%|▋| 73/100 [21:49<08:16, 18.2025-01-25 16:47:42,118 - INFO - Processing query: 4℃ ホールディングスの2024年2月29日現在の連結での従業員数は何名か。
2025-01-25 16:47:50,022 - INFO - API request completed in 5.19 seconds
2025-01-25 16:47:50,027 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164750.json
Processing queries:  74%|▋| 74/100 [21:57<06:36, 15.2025-01-25 16:47:50,028 - INFO - Processing query: カ ゴメの国内加工食品事業における通販部門の2023年度の売 上収益は何百万円ですか
2025-01-25 16:48:19,613 - INFO - API request completed in 27.69 seconds
2025-01-25 16:48:19,617 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164819.json
Processing queries:  75%|▊| 75/100 [22:27<08:08, 19.2025-01-25 16:48:19,618 - INFO - Processing query: ク レハの社員で生成AIを活用しているのは現在で約何割？   
2025-01-25 16:48:31,031 - INFO - API request completed in 9.63 seconds
2025-01-25 16:48:31,035 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164831.json
Processing queries:  76%|▊| 76/100 [22:38<06:50, 17.2025-01-25 16:48:31,036 - INFO - Processing query: キ ッツグループの企業の中で、「株式会社」という文字を除 き、ひらがな・カタカナ・漢字を全て含む会社名を答えよ 。（回答には株式会社も含めること）
2025-01-25 16:48:40,268 - INFO - API request completed in 8.58 seconds
2025-01-25 16:48:40,273 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164840.json
Processing queries:  77%|▊| 77/100 [22:47<05:39, 14.2025-01-25 16:48:40,274 - INFO - Processing query: 明 治グループが特定した12のマテリアリティのうち、ステー クホルダーにとっての重要度も明治グループにとっての重 要度も非常に高い2つを挙げてください。
2025-01-25 16:49:06,225 - INFO - API request completed in 25.43 seconds
2025-01-25 16:49:06,231 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164906.json
Processing queries:  78%|▊| 78/100 [23:13<06:38, 18.2025-01-25 16:49:06,232 - INFO - Processing query: ラ イフコーポレーションの2023年度実績において、都市ガス と重油ではどちらが多くCO2を排出しているか？
2025-01-25 16:49:29,996 - INFO - API request completed in 23.50 seconds
2025-01-25 16:49:30,002 - INFO - Saved result to M:\ML\signatejpx\ou2025-01-25 16:49:42,561 - INFO - API request completed in 12.26 seconds
2025-01-25 16:49:42,568 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_164942.json
Processing queries:  80%|▊| 80/100 [23:50<05:52, 17.2025-01-25 16:49:42,569 - INFO - Processing query: ハウス食品グループの海外重点エリ アをすべて挙げてください。
2025-01-25 16:50:01,090 - INFO - API request completed in 18.27 seconds
2025-01-25 16:50:01,096 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_165001.json
Processing queries:  81%|▊| 81/100 [24:08<05:40, 17.2025-01-25 16:50:01,097 - INFO - Processing query: 日清食品ホールディングスの社内専 用AIの名前は？
2025-01-25 16:50:18,641 - INFO - API request completed in 17.24 seconds
2025-01-25 16:50:18,646 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_165018.json
Processing queries:  82%|▊| 82/100 [24:26<05:20, 17.2025-01-25 16:50:18,647 - INFO - Processing query: 4℃ホールディングスの2024年2月期のEC売上高は何百万円か。
2025-01-25 16:50:39,953 - INFO - API request completed in 21.10 seconds
2025-01-25 16:50:39,957 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_165039.json
Processing queries:  83%|▊| 83/100 [24:47<05:20, 18.2025-01-25 16:50:39,958 - INFO - Processing query: 東洋エンジニアリングの独自開発の スケジュール最適化システムの名前は？
2025-01-25 16:50:48,872 - INFO - API request completed in 8.71 seconds
2025-01-25 16:50:48,876 - INFO - Saved result to M:\ML\signatejpx\output\results\result_20250125_165048.json
Processing queries:  84%|▊| 84/100 [24:56<04:13, 15.2025-01-25 16:50:48,877 - INFO - Processing query: 東洋エンジニアリングの取締役が有 する専門的知見や経験分野として多く選択されている項目は、『法務・法規則』と『人事・労務』のどちらでしょうか？
