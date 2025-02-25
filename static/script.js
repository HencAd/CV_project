const dropZone = document.getElementById('dropZone');
const videoPreview = document.getElementById('videoPreview');
const videoContainer = document.getElementById('videoContainer');
const uploadBtn = document.getElementById('uploadBtn');
const resultDiv = document.getElementById('result');
const containerDiv = document.getElementById('containerDiv');
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
	    containerDiv.innerHTML = '';
     });


uploadBtn.addEventListener('click', async () => {
    if (!file) return alert('You must upload a video first.');
    
    resultDiv.innerHTML = '<div class="loading-spinner"></div>';
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
	containerDiv.innerHTML = '<img src="/static/attention_heatmap.gif" alt="Heatmap">';

		  
    resultDiv.innerHTML = `Result: ${data.result} (Confidence: ${data.confidence}%)`;
});
