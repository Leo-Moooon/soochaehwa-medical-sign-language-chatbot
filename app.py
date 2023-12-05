# core
import os, sys
import re
from glob import glob

# flask
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# langchain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# torch
import torch

torch.set_default_dtype(torch.float64)

# utils
import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger().setLevel(logging.ERROR)

from dotenv import load_dotenv

load_dotenv()
container = os.environ.get('container')

# Modules
from modules.Inference import Predict

count = 0  # count 변수를 전역 변수로 초기화


def load_llm(model_opt='gpt3.5'):
    """

    * 기존 app.py - line30의 llm 선언에 해당하는 내용입니다.

    * 변경사항
      - api_key를 dotenv에서 받아오도록 했습니다.

    """
    api_key = os.environ.get('chatgpt_api_key')
    if model_opt == 'gpt3.5':
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         temperature=0,
                         openai_api_key=api_key,
                         streaming=True
                         )
    return llm


def init_qa_chain(llm):
    """

    * 기존 app.py - line38, 39의 qa_chain, qa_chain2 선언과 동일한 내용입니다.

    """
    db_dir = os.path.join(container, 'db')
    load_db = Chroma(persist_directory=db_dir, embedding_function=embeddings_model)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=load_db.as_retriever())
    qa_chain2 = RetrievalQA.from_chain_type(llm, retriever=load_db.as_retriever())
    return qa_chain, qa_chain2

def chat_qa(qa):
    """

    * 기존 app.py - line 67의 `def chat_qa()`와 동일한 함수입니다.

    """
    result_data = qa_chain({"query": qa})
    qa_result = result_data.get('result')
    return qa_result


# Set LLM, embedding model, langchain
llm = load_llm('gpt3.5')
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
qa_chain, qa_chain2 = init_qa_chain(llm)

def predict_data():
    video_path_list = glob(f'{container}/videos/**/*.mp4', recursive=True)
    video_path_list = sorted(video_path_list)
    device = torch.device('cpu')
    SLR = Predict(device)
    result = []
    for videopath in video_path_list:
        top1 = True
        word_pred = SLR.predict(videopath, top1)

        word = word_pred.split('_')[0]
        result.append(word)
    print(result)
    return result


def chat_qa(qa):
    result_data = qa_chain({"query": qa})
    res_qa = result_data.get('result')
    return res_qa


# Init Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(container, 'upload')
socketio = SocketIO(app)

count = 0  # initialize number for filename indexing


@app.route('/')
def splash():
    """

    최초에 2-3초 간 띄워진 후, 자동으로 채팅화면으로 넘어갑니다.

    """
    return render_template('splash.html')


@app.route('/chat')
def chat():
    """

    기존 app.py - line78에서의 엔드포인트 '/chat'에 해당하는 페이지입니다.

    """
    return render_template('chat.html')


@app.route('/record')
def record():
    """

    '/main'에서 녹화버튼을 누르면 이동되는 페이지입니다.

    """
    return render_template('record.html')


@app.route("/upload", methods=["POST"])
def print_data():
    global count  # count 변수를 전역 변수로 선언
    count += 1  # count를 1씩 증가시켜 사용

    uploaded_file = request.files['video']
    if uploaded_file:
        file_path = f'{container}/videos/' + \
                    str(count) + "_" + uploaded_file.filename
        uploaded_file.save(file_path)
        print("파일이 저장된 경로:", file_path)
        return jsonify({"확인": "파일 업로드 및 경로 확인 완료"})
    else:
        return jsonify({"에러": "파일이 업로드되지 않았습니다."})


def result_predict_query():
    res = []
    print('\033[38;5;208m' + f'''Predicting SL ...''' + '\033[0m')
    predict_result = predict_data()
    print('\033[38;5;208m' + f'''Predicted Result: {predict_result}''' + '\033[0m')

    question = '''
    1. 아래 List에 제시 단어를 조합하여 의문형 문장을 답변해줘, 
    2. 조합 문장은 최적의 문장 1개만 출력해줘
    3. 출력 문장은 아래 형태를 유지하여 만들어줘 
       (예시 : 제시 단어(사용자 제시 단어) > [여드름,치료법], 의문형 문장(GPT 답변 내용) > [여드름 치료법 알려주세요])
    4. 제시 단어(사용자 제시 단어) > ''' + str(predict_result)

    print('\033[38;5;208m' + f'''QA Chain ...''' + '\033[0m')

    result_data = qa_chain({"query": str(question)})
    print('\033[38;5;208m' + f'''result_data: {result_data}''' + '\033[0m')

    res_qa = result_data.get('result')
    res_qa2 = res_qa.split('\n')
    res_qa3 = ''
    try:
        res_qa3 = res_qa2[1].strip()
    except IndexError:
        res_qa3 = res_qa2[0].strip()
    except IndexError:
        res_qa3 = res_qa2


    # 정규식 표현
    pattern = r'\[([^[\]]+)\]'
    items = re.findall(pattern, str(res_qa3))

    # 최종 클라이언트 질문: items_query
    items_query = items[0].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
    print('\033[38;5;208m' + f'''items_query: {items_query}''' + '\033[0m')
    return items_query


def result_predict_answer(items_query):
    print('\033[38;5;208m' + f'''QA Chain2 ...''' + '\033[0m')

    fina_query = str(items_query) + " 항목을 나누어 300자 이내로 답변해줘"
    # 출력된 문장 질문
    result_data2 = qa_chain2({"query": fina_query})
    chat_final_result = result_data2.get('result')
    print(chat_final_result)
    print('\033[38;5;208m' + f'''chat_final_result: \n{chat_final_result}''' + '\033[0m')

    return chat_final_result


@socketio.on('message_from_client')
def handle_message(data):
    # Process the received message data
    client_message = data['message']

    # 클라이언트: 문장 송출
    emit('message_from_server',{'senderName': 'client', 'message': client_message}, broadcast=True)

    # ChatGPT 답변 생성 -> 서버 답변 출력
    server_message = result_predict_answer(client_message)
    emit('message_from_server', {'senderName': 'server', 'message': server_message}, broadcast=True)



@app.route('/result')
def handle_video():
    print('\033[38;5;208m' + f'''Recording finished.''' + '\033[0m')

    # 수어 변환 및 문장 변환 -> 클라이언트 질문 출력
    client_message = result_predict_query()
    socketio.emit('message_from_server', {'senderName': 'client', 'message': client_message})

    # ChatGPT 답변 생성 -> 서버 답변 출력
    server_message = result_predict_answer(client_message)
    socketio.emit('message_from_server', {'senderName': 'server', 'message': server_message})

    # 디렉토리 내 영상 모두 삭제

    return render_template('chat.html')



if __name__ == '__main__':
    socketio.run(app, debug=True, port='8888')