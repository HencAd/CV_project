document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const fileInput2 = document.getElementById("fileInput2");
    const dropArea = document.getElementById("dropArea");
    const buttonArea = document.getElementById("buttonArea");
    const statusCircle = document.getElementById("statusCircle");
    const waitText = document.getElementById("waitText");
    const highlightBar = document.getElementById("highlightBar");

    function openFileInput() {
        fileInput.click();
    }

    dropArea.addEventListener("click", openFileInput);
    buttonArea.addEventListener("click", openFileInput);

    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        handleFileUpload(files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        handleFileUpload(e.target.files[0]);
    });

    fileInput2.addEventListener("change", (e) => {
        handleFileUpload(e.target.files[0]);
    });

    function handleFileUpload(file) {
        if (!file) return;

        // Pokaż kropkę i tekst "Processing..."
        statusCircle.style.display = "block";
        statusCircle.classList.add("loading");
        waitText.style.display = "block";
        waitText.textContent = "Processing...";

        // Ukryj poprzednią linię, jeśli była widoczna
        highlightBar.classList.remove("content", "commercial");

        const formData = new FormData();
        formData.append("file", file);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            
            statusCircle.style.display = "none";
            waitText.style.display = "none";

            if (data.result === "content") {
                highlightBar.classList.add("content");  // Zielona linia
                waitText.textContent = `Content (${data.confidence}%)`;  
            } else {
                highlightBar.classList.add("commercial"); // Czerwona linia
                waitText.textContent = `Commercial (${data.confidence}%)`;
            }
        })
        .catch(error => {
            console.error("Error:", error);
            waitText.textContent = "Error processing file";
            statusCircle.style.display = "none";
        });
    }
});