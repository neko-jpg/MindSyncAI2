以下は、v3.0の骨子（Energy一貫化・LOCO・OSCR一次化・ストリーミング順序）を維持しつつ、いただいた「v3.1 改善カタログ」を全量反映して精度（Known精度×Open-set拒否×安定性）を一段押し上げる、実装可能な統合仕様です。運用面（署名JSON/ゲート/監視）にも整合しています。

英語SER・Open-set学習アルゴリズム v3.1（production-grade, boosted）

目次
- 0. 概要（差分ハイライトとv3.0→v3.1対応マップ）
- 1. タスク定義とKPI（Risk-Coverageを常設）
- 2. データ方針 v3.1（近傍OE=実+合成、話者/話法の層化）
- 3. 前処理・特徴・増強（条件付き処理/TTA on ambiguity/軽対抗）
- 4. モデル設計 v3.1（教師小アンサンブル/生徒=ハイブリッドプーリング＋Cond-Norm＋早期Abstain）
- 5. 損失・蒸留・OE v3.1（LDAM-CB/対比/スペクトル正則/合成近傍）
- 6. OOD検出 v3.1（Energy＋密度G-score＋Conformal τout）
- 7. 校正 v3.1（Dirichlet+CWT+Isotonic後詰め、オンライン分散重み付き微調整）
- 8. ストリーミング推論 v3.1（条件付きしきい値、HSMM化、曖昧時TTA、動的変化率）
- 9. 量子化/最適化 v3.1（Mixed-Precision、KLキャリブ、QAT蒸留=後校正）
- 10. 評価プロトコル v3.1（AURC/Excess-Risk、Conformal検証）
- 11. アブレーション/実験計画 v3.1
- 12. 実装・再現性・MLOps v3.1（署名JSON/ドリフト監視/Shadow-Deploy）
- 13. リスク・ガードレール（v3.1追加）
- 14. スケジュール v3.1（優先投入順とGo/No-Go）
- 付録A：環境別しきい値プロファイルの例
- 付録B：G-score/Conformal/HSMM/TTA/Laplace擬似コード
- 付録C：代表初期値（LOCO-devで再最適化）

0. 概要（差分ハイライトと対応マップ）
- 即効3本柱
  - 条件付きしきい値（τout/δ/H0/θc）を環境（SNR/帯域/T60/Overlap）で自動切替
  - 曖昧時のみTTA（微小ノイズ/シフト/速度）で平均化
  - HMM→HSMM化（クラス別持続分布）
- 中期の底上げ
  - Energy＋密度（Mahalanobis/kNN）でG-scoreを導入、Conformalでτoutに保証
  - 校正強化：Dirichlet＋Class-wise Temperature（CWT）、Isotonicを後詰め
  - 生徒最終層のLaplace（近似ベイズ）で分散を得て確信の健全化
- モデル/学習の底堅さ
  - 教師の軽アンサンブル（WavLM＋HuBERT等）で境界を滑らかに蒸留
  - 生徒プーリングをNetVLAD-lite＋統計のハイブリッド、Cond-Normで環境頑健化
  - 近傍OEを“実＋合成（TTS/VC/韻律変調）”で強化、LDAM-CB/微量SupCon/スペクトル正則
- 量子化の信頼性
  - Mixed-Precisionポリシー＋KLキャリブ＋QAT蒸留（後校正KD）でINT8でもAECE/OSCR維持
- 監視・運用
  - Risk-Coverage（AURC/Excess-Risk）とEnergy/G-scoreのPSI/Wassersteinを週次監視
  - 署名JSONにConformal τout/しきい値プロファイル/CWT/密度統計も統合

対応マップ（v3.0→v3.1）
- Energy一貫化：維持（pre-calib固定）。G-scoreの密度もz_pre由来表現で一貫
- Unknown判定：Energy一段→G-score→校正→top-2/entropy→HSMM→変化率→θc
- LOCO運用：Conformal τoutや密度パラメータもLOCO-devで確定→test固定
- 署名運用：threshold_profile.json, conformal.json, density_stats.json, calib.json を一元署名

1. タスク定義とKPI（Risk-Coverage常設）
- 既存KPIを維持＋追加
  - Known分類：Macro-UAR/Macro-F1/AECE/MCE/Classwise-ECE
  - Unknown検出：AUROC/AUPR-Out/FPR95/AUPR-In
  - Open-set分類：OSCR/CCR@γ/H-mean
  - 安定性：TTFC、切替回数/分、平均持続、フレーム間JSD
  - 選択的分類：AURC（Area under Risk-Coverage）/Excess-Risk（拒否含む）
- ゲート（Go/No-Go）にAURC/Excess-Riskを追加（運用の拒否戦略健全性）

2. データ方針 v3.1（近傍OE=実＋合成、層化の強化）
- 近傍OEの拡充
  - 実：boredom/contempt/confusion/excited/amused/frustration、ささやき/早口/鼻音/ため息/重畳など（既存）
  - 合成：TTS/VCで同話者・同テキスト・韻律/速度/息継ぎ/囁き変調→Unknown扱い（近傍係数を大）
- 遠方OE：非音声（家電/環境/機械）を環境カテゴリ別に追加
- 話者/話法の層化：性別/年代/方言/話法をメタ属性に→層化サンプリング（train/dev/test全域）
- LOCO常設：3-way（CREMA-D / IEMOCAP / MSP-Podcast）＋言語外は別枠報告
- 配備プロファイル：SNR/T60/帯域ヒストグラム→AugMixサンプラー/Cond-Norm埋め込みに反映

3. 前処理・特徴・増強 v3.1（条件付き処理/対抗/TTA）
- 条件付き前処理
  - 可聴域外対策：帯域外（>8 kHz）をLPFで抑制、超音波入力の異常を遮断
  - 8/16 kHz帯域を自動検出しフィルタ設定
- 特徴：生徒=80-dim log-mel（CMVN/PCENはA/B）＋dither、教師=16 kHz波形（SSL）
- 増強
  - Stratified AugMix（配備分布に整合、AECE劣化≤1%制約）
  - 軽対抗（spec域の小PGD、ε小）を5–10%バッチに混入
  - SpecAug/VTLP/速度/ピッチはv3.0準拠
- TTA-on-ambiguity（推論）
  - p1−p2<δ or H(p)>H0 の暫定Unknownに限り、微小ノイズ/±時刻シフト/±速度で2–3サンプル平均化（因果性維持）

4. モデル設計 v3.1（教師アンサンブル/生徒強化）
- 教師（学習時のみ）
  - 例：WavLM-Base＋HuBERT-Base を弱アンサンブル（logit/表現の平均＋コサイン蒸留）
  - Multi-task：VA回帰ヘッドは維持（表現滑らか化）
- 生徒（~2–3M, INT8耐性）
  - 干渉少なめConv/TDNN/TCNベース、Attentionなし
  - プーリング：NetVLAD-lite（少クラスタ）＋統計プーリングのハイブリッド（連結→1×1圧縮）
  - Cond-Norm：SNR/Overlap/帯域/T60を小埋め込み→LayerNormのγ/βに注入（微調整）
  - 早期Abstainヘッド：軽量前段でEnergyのみ即時判定（早期Unknown出し）
  - 最終層ベイズ：最後のFCにLaplace近似（後付け）で予測分散を取得（校正/Conformal/オンライン微調整の重みに使用）
  - 量子化耐性：LayerNorm/1×1に集約、LogitNormは弱項維持

5. 損失・蒸留・OE v3.1
- 分類：Class-balanced CE＋LDAM（小）or Focal-Tversky（弱）で希少クラスUAR↑（ls=0.05は維持）
- 蒸留：KL(teacher→student, T=2–4, α=0.5)＋表現コサイン（λcos 0→0.5）＋終盤MSE微小
- SupCon（微量）：τ=0.1–0.2、同クラス近接/異クラス疎遠でtop-2 margin↑
- One-vs-Rest補助：λ_ovr=0.05–0.1（維持）
- 正則：LogitNorm（1e-4）＋最後層/近辺に軽スペクトル正則（スペクトルノルム制御）
- OE（pre-calib Energy一貫/マルチマージン/カリキュラム）
  - 実近傍：合成近傍＝実近傍を50:50で混在（終盤でもOE≤50%）
  - m_out = m0 + γ·hardness（近傍γ↑、遠方γ↓）、T_Eは外部foldでロック
- 逆伝播の安定化：grad clip=5.0、AMP、SWA/seed平均は継続

6. OOD検出 v3.1（Energy＋密度＋Conformal）
- 密度推定（中間表現h）
  - Mahalanobis：クラス中心μk/共分散ΣkをShrinkage（Ledoit-Wolfなど）でLOCO-devから推定
  - 代替：小kNN（FAISSライト/コサイン距離）
- G-score
  - E(x) = −T_E logsumexp(z_pre/T_E)
  - D(x) = min_k (h(x)−μk)^T Σk^-1 (h(x)−μk)
  - G(x) = α·(−E(x)) + β·(−D(x))（α/βはLOCO-devで最適化）
- しきい値：Conformal（LOCO-dev KnownのG分布の上側分位 1−q）でτoutを設定（保証付き）
  - 既存の外部fold最適化と整合：Conformal τoutをベース（保証）、運用では±微調整（署名管理）
- ストリーミング順序に統合：Energy→G-score→校正…の二段（詳細は8章）

7. 校正 v3.1（Dirichlet+CWT、Isotonic後詰め）
- オフライン校正（Knownのみ、5-fold CV）
  - Dirichlet校正（推奨）の前段にClass-wise Temperature（対角）を置き、L2で過適合抑制
  - 代替：Matrix/Vector。選択基準=AECE/MCE/Classwise-ECE/OSCR（LOCO-dev）
- バックアップ校正：Isotonic/Splineの単調校正を保存（過適合に強い後詰め）
- オンライン安全再校正
  - 高確信Known（p>0.9）のみに対しb（バイアス）最小二乗で微調整
  - 重み=予測分散（Laplace）で安全化（ノイズ時の過補正回避）
- 署名保存：calib.jsonにCWT/Dirichletパラメータ、isotonicバックアップ、AECE/MCE/CIを記録

8. ストリーミング推論 v3.1（適用順序）
- 事前：環境推定（SNR/帯域/T60/Overlap）→env_idを決定
- 順序（0.5 s hop, 1.5–2.0 s因果窓）
  1) VAD→バッファ、Overlap推定
  2) 生徒（z_pre, h）、早期Abstain：E_pre≥τout(env) なら即時Unknown
  3) 密度（D）→G-score、G≥τout(env, conformal) ならUnknown
  4) 校正（Dirichlet+CWT）→p
  5) 暫定ゲート：p1−p2<δ(env) or H(p)>H0(env) → 暫定Unknown
     - 該当時のみTTA（2–3通り）でpを平均し再評価
  6) HSMM（クラス別最小持続＋持続分布）→HMM遷移→変化率ペナルティ
     - 直近の暫定Unknown率/Overlap率が高いとλ_chg↑（混雑時の自動抑揺れ）
  7) 薄閾値θc(env)を適用、Overlap高（p_ov>τ_ov）はUnknown優先
- 環境別プロファイル：threshold_profile.jsonからロード（署名）

9. 量子化/最適化 v3.1（INT8でも校正を崩さない）
- Mixed-Precision
  - 前段Conv/FC=INT8（w: per-channel, a: per-tensor）
  - プーリング直後と最終FC手前=FP16 or W-INT8/A-FP16（校正保持のため）
- PTQのKLキャリブ
  - Actクリップ範囲をKL最小で最適化（Energy/Gの上尾を温存）
  - キャリブは層化2000+セグメント（Known/遠方/近傍×SNR×RIR×話者）
- Gate（PTQ→QAT移行）指標を拡張
  - UAR>−3pp, AECE<+3%, FPR95<+3pp, AUROC>−1.5pp, OSCR>−2pp に加え AURC/Excess-Riskの悪化上限を設定
- QAT（8–10 epoch）
  - 教師の“後校正出力”に対してKD（分類＋校正の同時蒸留）→INT8後のAECE/OSCR回復
  - QAT後にcalib再学習＋τout/G-score/Conformal再最適化

10. 評価プロトコル v3.1
- 既存に加え：
  - AURC/Excess-Risk、Conformalカバレッジ検証（期待超過率≤q+ε）
  - 環境別（env_id別）でOSCR/FPR95/TTFCを分割報告
- LOCO：Calib/CWT/密度/Conformal/τout/envプロファイルはdevで確定→test固定
- 3 seeds＋95%CI、Shadow-Deploy時はRisk-Coverage曲線のJSD/KS統計で一致度を監視

11. アブレーション/実験計画 v3.1
- 即効群：条件付きしきい値、TTA-on-ambiguity、HSMM（三者の単独/併用）
- OOD群：G-score（E vs E+D/kNN）、Conformal q（0.01/0.02/0.05）
- 校正群：Dirichlet vs Dirichlet+CWT vs Matrix/Vector、Isotonic後詰め有無
- 学習群：LDAM-CB有無、SupCon係数、スペクトル正則有無、合成近傍比率（0/25/50%）
- モデル群：NetVLAD-lite有無、Cond-Norm有無、Laplace有無、教師アンサンブル有無
- 量子化群：Mixed-Precisionポリシー、KLキャリブ有無、QAT蒸留（後校正）有無
- 指標：OSCR/H-mean/FPR95/AURC/AECE/TTFC/切替回数、環境別/LOCO別

12. 実装・再現性・MLOps v3.1
- 署名付きアーティファクト
  - calib.json（CWT/Dirichlet/Isotonic）、threshold_profile.json（env→τout/δ/H0/θc）
  - density_stats.json（μk/Σk/αβ）、conformal.json（q/τout/保証CI）
  - 既存のenergy_space/T_E/m0/γも併記、一元署名
- 監視
  - 週次：OSCR/CCR/H-mean、FPR95、AECE/AURC、Unknown率（近傍/遠方/非音声/重畳）
  - ドリフト：Energy/G-scoreのPSI/Wasserstein、Risk-Coverage曲線のJSD
- 運用
  - OEバッファ週次（誤確信Top-K）＋合成近傍の月次更新
  - Calib/Conformal再学習はモデル不変のままホットスワップ（署名更新）
  - Shadow-Deployでオン/オフライン曲線の一致度をチェック

13. リスク・ガードレール
- 近傍OE過学習：OE≤50%、OSCR×Known-UARのPareto監視、γ_nearを自動弱化
- 密度スケール崩れ：Shrinkage/ロバスト共分散（最小固有値制約）、クラス毎正則
- TTAレイテンシ：曖昧率が上限超過でTTA頻度/数を制限
- しきい値過調整：Conformal τoutを下限保証として維持、運用調整は±小幅
- 早期Abstainの過拒否：環境良条件では閾値を緩めTTFCを確保（プロファイルで切替）

14. スケジュール v3.1（優先投入順）
- Week 1
  - 環境推定器＋条件付きしきい値、TTA-on-ambiguity、HSMM実装→LOCO-devで再最適化
  - CalibをDirichlet+CWTへ更新、Isotonic保存、Conformal τout導入
- Week 2
  - G-score（Mahalanobis）導入、密度推定/署名JSON整備
  - 生徒：プーリングハイブリッド＋Cond-Norm、最終層Laplace（後付け）
  - PTQ: KLキャリブ、Gate→必要ならQAT（後校正蒸留）
- Go/No-Go：OSCR/H-mean/FPR95/AECE/AURC が目標域＋Conformal保証達成

付録A：threshold_profile.json（例）

{
  "meta": {"hash":"...", "timestamp":"...", "signature":"..."},
  "env_buckets": {
    "clean_16k_lowOv": {"snr":[25,35], "t60":[0.1,0.3], "band":"16k", "ov":[0,0.1]},
    "noisy_8k_midOv": {"snr":[5,15],  "t60":[0.4,0.7], "band":"8k",  "ov":[0.1,0.4]}
  },
  "params": {
    "clean_16k_lowOv": {"tau_out":2.35, "delta":0.05, "H0":1.4, "theta":{"angry":0.08,"sad":0.12}, "tau_ov":0.3},
    "noisy_8k_midOv": {"tau_out":2.60, "delta":0.10, "H0":1.7, "theta":{"angry":0.12,"sad":0.15}, "tau_ov":0.25}
  }
}

付録B：擬似コード（抜粋）

- G-score/Conformal

# 推定（LOCO-dev, Known）
mu_k, Sigma_k = fit_shrinkage_gaussians(h_dev, y_dev)
alpha, beta = grid_search_alpha_beta(E_dev, D_dev, target=OSCR)
G_dev = alpha*(-E_dev) + beta*(-D_dev)
tau_out = quantile(G_dev, 1 - q)  # Conformal保証
save_json({mu_k,Sigma_k,alpha,beta,tau_out,q})

# 推論
z, h = student(window)
E = -T_E * logsumexp(z/T_E)
D = min_k (h-mu_k)^T inv(Sigma_k) (h-mu_k)
G = alpha*(-E) + beta*(-D)
if G >= tau_out_env: emit('Unknown'); continue

- HSMM（最小持続＋持続分布）

y_tmp = gate(p)  # 暫定
state = hsmm.update(y_tmp)  # クラス別 min_dur と duration PMF（幾何/ガンマ）
y_hmm = hmm_smoother.update(state)
y_out = change_rate_penalty.update(y_hmm, lambda_chg(env, congestion))

- TTA-on-ambiguity

p = calibrator(z)
if (top1(p)-top2(p) < delta_env) or (entropy(p) > H0_env):
    P = [p]
    for aug in [shift_small, add_small_noise, speed_0.98]:
        z_a = student(aug(window))
        P.append(calibrator(z_a))
    p = mean(P)

- Laplace（最後層）

# 学習後にフィット
H = approximate_hessian_last_layer(train_loader)  # Gauss-Newton近似
Sigma_w = (H + λI)^-1
# 推論時：予測分散から温度補正 or オンライン校正の重み
var_pred = phi(x)^T Sigma_w phi(x)

- 条件付きしきい値選択

env = bucketize(SNR, T60, Overlap, Band)
params = profile[env]  # τout, δ, H0, θc, τ_ov, λ_chg_level

付録C：代表初期値（LOCO-devで再最適化）
- T_E=2–4（pre）、q=0.02（Conformal）
- α=1.0, β=0.5（初期）、Shrinkage=0.1–0.3
- HSMM最小持続：{angry:0.6s, surprise:0.5s, neutral/sad:1.0s, others:0.75s}
- 変化率コストλ_chg：clean=0.2, noisy/overlap-high=0.4
- TTA：N=2–3、追加RTF<10%（曖昧時のみ）
- OE：m0_far≈τout−0.2, m0_near≈τout+0.1, γ_far=0.3, γ_near=0.8（維持）

期待効果（経験則）
- FPR95：Energy単独→G-scoreで1–3pp低下（近傍優位）
- AECE：Dirichlet+CWTで0.5–1.0%改善、QAT蒸留（後校正）でINT8後の崩れを回避
- OSCR/H-mean：合成近傍OE＋SupCon（弱）で1–2pp上昇
- TTFC：HSMM＋TTA-on-ambiguityで0.05–0.15 s短縮（条件依存）

最後に
- v3.1は、v3.0の「Energy一貫・LOCO・順序仕様」を崩さずに、曖昧域での確信制御（TTA/HSMM/条件付きしきい値）と、近傍OODの取りこぼし削減（G-score/Conformal）を同時に達成します。
- まずは「条件付きしきい値→TTA-on-ambiguity→HSMM→Dirichlet+CWT→Conformal τout」の順で導入し、次に「G-score→Laplace→Cond-Norm→合成近傍」を加えると、開発コストと効果のバランスが良いです。
- すべての新規パラメータ/統計/しきい値は、既存の署名JSON運用に統合し、Shadow-DeployでRisk-Coverageの一致を確認してから本番ホットスワップしてください。

