let mediaRecorder;
let recordedChunks = [];

// Access the webcam video
async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('webcam-preview').srcObject = stream;
    return stream;
}

// Show preview
async function previewWebcam() {
    const stream = await startWebcam();
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    return stream
}

// Start recording
async function startRecording() {
    const stream = await previewWebcam()

    recordedChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
}

// Save recording
function saveRecording() {
    console.log("스타트리코딩 함수 실행");
    if (recordedChunks.length === 0) {
        console.error('저장할 녹화된 청크가 없습니다.');
        return;
    }

    const blob = new Blob(recordedChunks, { type: 'video/mp4' });

    // FormData 객체를 생성하여 블롭을 전송합니다.
    const formData = new FormData();
    formData.append('video', blob, 'recording.mp4');

    // 최신 브라우저에는 XMLHttpRequest 대신 Fetch API 사용
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            console.log('비디오가 성공적으로 업로드되었습니다.');
        } else {
            console.error('비디오 업로드 실패');
        }
    })
    .catch(error => console.error('페치 중 오류 발생:', error));
}


// Stop recording
function stopRecording() {
    saveRecording();
    console.log("스탑리코딩 함수 실행");
    mediaRecorder.stop();

    // 'startRecording' 함수를 여기에서 호출하지 않도록 변경
    // fetch('/result') 호출을 통해 서버로 결과를 전송하도록 함

    // Use Fetch API to send a request to the server
    fetch('/result')
        .then(response => response.json())
        .then(data => {
            console.log(data); // You can handle the response data here
        })
        .catch(error => console.error('Error during fetch:', error));
}


function switchDisplay(selector, option) {
    document.querySelector(selector).style.display = option
}

function popupMessageInRecord(selector, message, delay) {
    delay *= 1000
    document.querySelector(selector).innerHTML = message;
    document.querySelector(selector).style.display = 'block';
    setTimeout(function() {
        document.querySelector(selector).style.display = 'none';
}, delay);
}

// Event listeners
document.addEventListener('DOMContentLoaded', previewWebcam);

document.getElementById('btn-start').addEventListener('click', () => {
    startRecording();

    switchDisplay('#btn-start',         'none');
    switchDisplay('.text-start-record', 'none');
    switchDisplay('#btn-save',          'block');
    switchDisplay('.text-save-record',  'block');
    switchDisplay('#btn-stop',          'inline');
    switchDisplay('#btn-reset',         'inline');
});

document.getElementById('btn-save').addEventListener('click', () => {
    mediaRecorder.stop();
    mediaRecorder.onstop = () => {
        saveRecording();
        startRecording();
    };

    popupMessageInRecord('.popup-message-content', '저장되었습니다!', 2);
});

document.getElementById('btn-stop').addEventListener('click', () => {
    stopRecording();
    popupMessageInRecord('.popup-message-content', '녹화가 완료되었습니다!', 2);
    setTimeout(function() {
        switchDisplay('#btn-start',         'block');
        switchDisplay('.text-start-record', 'block');
        switchDisplay('#btn-save',          'none');
        switchDisplay('.text-save-record',  'none');
        switchDisplay('#btn-stop',          'none');
        switchDisplay('#btn-reset',         'none');
    }, 2000)
});