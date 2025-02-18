const dropZone = document.getElementById('dropZone');
const videoPreview = document.getElementById('videoPreview');
const videoContainer = document.getElementById('videoContainer');
const uploadBtn = document.getElementById('uploadBtn');
const resultDiv = document.getElementById('result');
let file;

// Obsługuje przeciąganie pliku
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.border = '2px solid blue';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.border = '2px dashed #000';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    file = e.dataTransfer.files[0];
    if (file.type.startsWith('video/')) {
        console.log("File detected: ", file);
        const url = URL.createObjectURL(file);
        console.log
        videoPreview.src = url;
        videoPreview.style.display = 'block'; // Wideo staje się widoczne
        videoContainer.style.display = 'block'; // Pokazanie kontenera wideo
        dropZone.style.display = 'none'; // Ukrycie dropzone
    }
});

// Obsługa przycisku upload
uploadBtn.addEventListener('click', async () => {
    if (!file) return alert('Najpierw załaduj plik wideo.');
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    resultDiv.innerHTML = `Result: ${data.result} (Confidence: ${data.confidence}%)`;
});