const dropZone = document.getElementById('dropZone');
const videoPreview = document.getElementById('videoPreview');
const videoContainer = document.getElementById('videoContainer');
const uploadBtn = document.getElementById('uploadBtn');
const resultDiv = document.getElementById('result');
const clearBtn = document.getElementById('clearBtn');

let file;


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
        videoPreview.style.display = 'block'; 
        videoContainer.style.display = 'block'; 
        dropZone.style.display = 'none';
    	clearBtn.style.display = 'block'; 
    }
});


clearBtn.addEventListener('click', () => {
	    videoPreview.src = '';
	    videoPreview.style.display = 'none';
	    videoContainer.style.display = 'none';
	    dropZone.style.display = 'flex';
	    clearBtn.style.display = 'none'; 
	    resultDiv.innerHTML = '';
     });


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
