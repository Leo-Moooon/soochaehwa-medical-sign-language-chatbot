const myName = "client";
const socket = io.connect('http://' + document.domain + ':' + location.port);

function init() {
    // Enter 키 이벤트
    $('div.input-div textarea').on('keydown', function(e) {
        if (e.keyCode == 13 && !e.shiftKey) {
            e.preventDefault();
            const message = $(this).val();

            // 메시지 전송
            sendMessage(message);
            // 입력창 비우기
            clearTextarea();
        }
    });
}

function createMessageTag(LR_className, senderName, message) {
    // 형식 가져오기
    let chatLi = $('div.chat-format ul li').clone();

    // 값 채우기
    chatLi.addClass(LR_className);
    chatLi.find('.senderName span').text(senderName);
    chatLi.find('.message span').text(message);

    return chatLi;
}

// 메시지 태그 append
function appendMessageTag(LR_className, senderName, message) {
    const chatLi = createMessageTag(LR_className, senderName, message);

    // $("div.chat:not(.format) ul").append(chatLi);
    $("div.chat ul").append(chatLi);

    // 스크바 아래 고정
    $('div.chat').scrollTop($('div.chat').prop('scrollHeight'));
}

function sendMessage(message) {
    const data = {
        'senderName': 'client',
        'message': message
    };
    // 'message_from_client' 이벤트를 서버로 emit
    socket.emit('message_from_client', data);
}

function clearTextarea() {
    $('div.input-div textarea').val('');
}

// 서버로부터 수신한 메시지 처리
socket.on('message_from_server', function(data) {
    const LR = (data.senderName !== myName) ? 'left' : 'right';
    appendMessageTag(LR, data.senderName, data.message);
});

// 초기화 함수 호출
init();