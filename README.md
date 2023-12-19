<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" type="text/css" href="templates/readme.css" />
    </head>
    <body>
        <h1>수채화 - 의료분야 수어 챗봇 서비스</h1>
        <div class="award">
            <h3>제 5기 KDT 해커톤 우수상(고용노동부 장관상) 수상</h3>
            <img src="src/images/award.jpeg" alt="award.jpeg">
        </div>
        <h2>프로젝트 개요</h2>
        <table class="summary">
            <tr>
              <th>기간</th>
              <td>2023.08.14 ~ 2023.11.28 (총 107일)</td>
            </tr>
            <tr>
              <th>인원구성</th>
              <td>문성우 박태휘 채성혁</td>
            </tr>
            <tr>
              <th>프로젝트 목표</th>
              <td>수어 인식이 가능한 챗봇 서비스의 개발을 통한 청각장애인의 의료격차 해소</td>
            </tr>
            <tr>
              <th>프로젝트 내용</th>
              <td>▪ 수어인식 모델 개발 간 GRB 비디오 입력과 keypoint 입력 시의 성능 비교<br>
                  ▪ 챗봇 개발 간 RAG(Retrieval-augmented generation, 검색 증강 생성) 기술의 적용을 통한 답변의 정확성, 신뢰성 향상</td>
            </tr>
            <tr>
              <th>개발환경 <br> (클라우드)</th>
              <td>  <code>CLOUD</code> Google Colab Pro+<br>
                    <code>OS</code>  Linux(Ubuntu 18.04.6 LTS) <br> 
                    <code>CPU</code> Intel Xeon <br> <code>GPU</code> V100 / A100 <br> 
                    <code>RAM</code> 40GB</td>
            </tr>
            <tr>
              <th>사용 언어 및 <br> 기술 스택</th>
              <td>  <code>Language</code>   <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/Javascript-F7DF1E?style=flat-square&logo=javascript&logoColor=white"/>
                                            <br> 
                    <code>Editor</code>     <img src="https://img.shields.io/badge/VS Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/Pycharm-000000?style=flat-square&logo=pycharm&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
                                            <br> 
                    <code>Modeling</code>   <img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/HuggingFace-f8a700?style=flat-square&logo=huggingface&logoColor=white"/>
                                            <br> 
                    <code>CV</code>         <img src="https://img.shields.io/badge/OpenCV(python)-412991?style=flat-square&logo=opencv&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/Albumentations-d90000?style=flat-square&logo=albumentations&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/TorchVision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
                                            <br> 
                    <code>NLP</code>        <img src="https://img.shields.io/badge/ChatGPT-0f9e7b?style=flat-square&logo=openai&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/LangChain-b3d053?style=flat-square&logo=langchain&logoColor=white"/>
                                            <br> 
                    <code>DB</code>         <img src="https://img.shields.io/badge/ChromaDB-307af8?style=flat-square&logo=chroma&logoColor=white"/>
                                            <br> 
                    <code>Web</code>        <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/AJAX-3087c5?style=flat-square&logo=ajax&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/JQuery-0769AD?style=flat-square&logo=jquery&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white"/>
                                            <br>
                    <code>Logging</code>    <img src="https://img.shields.io/badge/Weights & Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=white"/>
                                            <br> 
                    <code>Version Control</code>        <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/>
                                            <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"/>
                                            <br>
              </td>
            </tr>
        </table>
        <div id="table-of-contents">
            <h2>목차 및 바로가기</h2>
            <ol class="project-ul" start="1">
                <li>서비스</li>
                <ul class="serve-ul">
                    <li><a href="#testing">활용 예시(시연 영상)</a></li>
                    <li><a href="#service-ui">서비스 UI</a></li>
                    <li><a href="#service-architecture">서비스 구조도</a></li>
                </ul>
                <br>
                <li>데이터</li>
                <ul>
                    <li><a href="#data-section">데이터 수집</a></li>
                </ul>
                <br>
                <li>모델링</li>
                <ul class="modeling-ul">
                    <li><a href="#SSLR-V1">수어인식 모델: SSLR V1</a></li>
                    <li><a href="#SSLR-V2">수어인식 모델: SSLR V2</a></li>
                        <ul class="SSLRV2-ul">
                            <li><a href="#strategy01">성능 개선 전략 01: Keypoint 추출</a></li>
                            <li><a href="#strategy02">성능 개선 전략 02: Customized Normalization</a></li>
                        </ul>
                    <li><a href="">성능 비교: SSLR V1 vs V2</a></li>
                </ul>
                <br>
                <li>챗봇</li>                
                <ul class="chatbot-ul">
                    <li><a href="">사전 준비: Vector Database 구축</a></li>
                    <li><a href="">질문 가공: ChatGPT API, One-Shot Prompting을 통한 말뭉치 -> 문장 변환</a></li>
                    <li><a href="">챗봇 답변 반환: Langchain을 통한 ChatGPT 기반의 RAG(Retrieval-Augmented Generation) 구현</a></li>
                </ul>
                <br>
                <br>
                <li>추후 개선 방향</li>
            </ol>
        </div>
        <br>
        <div id="service-section">
            <h2>1. 서비스</h2>
            <h3 id="testing">활용 예시</h3>
            <img src="./src/images/service_testing.gif" height="400">
            <br>
            <br>
            <h3 id="service-ui">서비스 UI</h3>
            <img src="./src/images/service_ui.png" height="300" alt="service_ui.png">
            <br>
            <br>
            <h3 id="service-architecture">서비스 구조도</h3>
            <img src="./src/images/service_architecture.png" height="500" alt="service_architecture.png">
            <br>
            <br>
        </div>
        <br>
        <div id="data-section">
            <h2>2. 데이터</h2>
            <h3 id="data-section">데이터 수집</h3>
            <p>AI Hub - "수어 영상" 데이터셋에서 단어 영상 및 라벨 데이터 취득.</p>
            <ul>
                <li>EDA를 통해 전체 의료 용어 5,485건, 일상어 12,317건을 선별.</li>
                <li>데이터셋에서 제시하는 기준에 맞춰 Train 및 Validation 데이터 구분.</li>
                <li>제공된 Train과 Validation 데이터의 유사성이 높다고 판단하여, Test 데이터는 별도로 촬영하여 소량 수집하였음.</li>
            </ul>
        </div>
        <br>
        <div id="modeling-section">
            <h2>3. 모델링</h2>
            <h3 id="SSLR-V1">수어인식 모델: SSLR V1 (Soochaehwa Sign Language Recognizer V1)</h3>
            <table class="sslrv1-table">
                <tr>
                    <td width="300"><img src="./src/images/SSLRV1_full.png" height="300" alt="SSLRV1_full.png"></td>
                    <td>
                        <ol>
                            <li>접근 가설</li>
                            <ul>
                                <li>수어는 연속된 동작이므로, 행동을 인식하고 분류하는 문제로 파악</li>
                                <li>RGB 비디오를 학습하여 각 동작에 대한 일반화 기대</li>
                            </ul>
                            <br>
                            <li>아키텍쳐 특징</li>
                            <ul>
                                <li>사전학습된 DenseNet-121을 통해 각 프레임 내 특징 추출</li>
                                <li>Transformer-Encoder 구조로, 프레임 간 전후맥락을 고려한 학습 의도</li>
                            </ul>
                            <br>
                            <li>참고 자료</li>
                            <ul>
                                <li><em>"Beyond Short Snippets: Deep Networks for Video Classification"(Ng et al., 2015)</em></li>
                                <li><em>"Attention Is All You Need" (Vaswani, A., et al., 2017)</em></li>
                            </ul>
                        </ol>
                    </td>
                </tr>
            </table>
            <br>
            <h3>SSLR V1 학습 및 예측 결과</h3>
            <table>
                <tr>
                    <th>Accuracy</th>
                    <th>예측 결과</th>
                </tr>
                <tr>
                    <td><img src="./src/images/SSLRV1_accuracy_plot.png" width="350" alt="SSLRV1_accuracy_plot.png"></td>
                    <td><img src="./src/images/SSLRV1_predict_result.png" width="400" alt="SSLRV1_predict_result.png"></td>
                </tr>
            </table>
            <h4>원인 분석</h4>
            <li>부족한 특징 추출로 인한 underfit 현상.</li>
            <li>모델이 학습데이터에 과하게 편향되어 실제 데이터를 제대로 맞추지 못하는 것으로 추정.</li>
            <br>
            <h3 id="SSLR-V2">수어인식 모델: SSLR V2 (Soochaehwa Sign Language Recognizer V2)</h3>
            <table class="sslrv2-table">
                <tr>
                    <td width="300"><img src="./src/images/SSLRV2_full.png" height="300" alt="SSLRV2_full.png"></td>                    <td>
                        <ol>
                            <li>접근 가설</li>
                            <ul>
                                <li>많은 데이터를 필요로 하는 Vision Transformer의 특징이 
                                    SSLR V1에서 과소적합(underfit)의 원인이 되었을 것으로 추정</li>
                                <li>신체 키포인트 추출을 통해 특징 추출 극대화</li>
                            </ul>
                            <br>
                            <li>아키텍쳐 특징</li>
                            <ul>
                                <li><strong>특징 추출기 변경</strong>
                                    <ol>
                                        <li>RGB 비디오에서 신체 Keypoint 데이터 추출</li>
                                        <li>별도의 정규화 (Customized Normalization) 진행</li>
                                    </ol>
                                </li>
                                <li>BERT의 구조를 차용하되, 각 block 사이에 Batch Normalization 레이어 추가</li>
                            </ul>
                            <br>
                            <li>참고 자료</li>
                            <ul>
                                <li><em>"Preprocessing for Keypoint-Based Sign Language Translation without Glosses"(Kim & Baek, 2023)</em></li>
                                <li><em>"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." Devlin, J., et al. (2019).</em></li>
                            </ul>
                        </ol>
                    </td>
                </tr>
            </table>
            <h3 id="strategy01">성능개선 전략 01: Keypoint 추출</h3>
            <p>Mediapipe 라이브러리를 이용한 신체 Keypoint 추출. 그 후 데이터프레임으로 정규화</p>
            <img src="./src/images/SSLRV2_mediapipe.png" alt="">
            <br>
            <h3 id="strategy02">성능개선 전략 02: Customized Normalization</h3>
            <p>하단의 레퍼런스에서 제안한 Customized Normalization 기법을 보유 데이터셋에 적합하게 일부 수정하여 코드 구현 및 적용.</p>
            <table>
                <tr>
                    <th>STEP01: 하반신 제거 및 지역 분할</th>
                    <th>STEP02: 각 지역별 거리 정규화</th>
                </tr>
                <tr>
                    <td><img src="./src/images/SSLRV2_cn_step01.png" alt="SSLRV2_cn_step01.png"></td>
                    <td><img src="./src/images/SSLRV2_cn_step02.png" alt="SSLRV2_cn_step02.png"></td>
                </tr>
            </table>
            <li>레퍼런스: <em>"Preprocessing for Keypoint-Based Sign Language Translation without Glosses"(Kim & Baek, 2023)</em></li>
            <br>
            <h3>Ablation Study: 성능개선 전략 적용 여부에 따른 SSLR 성능 변화</h3>
            <img src="./src/images/SSLRV2_ablation_study.png" alt="SSLRV2_ablation_study.png">
        </div>
        <br>
        <div id="chatbot-section">
            <h2>4. 챗봇</h2>
            <h3>사전 준비: Vector Database 구축</h3>
            <h3>질문 가공</h3>
            <h3>챗봇 답변 반환</h3>
        </div>
        <br>
        <div id="future-improvements">
        <h2>5. 추후 개선 방향</h2>
        </div>
    </body>
</html>
