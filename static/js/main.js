document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileNameDisplay = document.getElementById('fileName');
    const uploadForm = document.getElementById('uploadForm');
    const loadingOverlay = document.getElementById('loadingOverlay');

    if (dropZone) {
        // Drag over
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        // Drag leave
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });

        // Drop
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                updateFileInfo();
            }
        });

        // Click to browse
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        // File input change
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                updateFileInfo();
            }
        });
    }

    function updateFileInfo() {
        const file = fileInput.files[0];
        if (file) {
            fileNameDisplay.textContent = file.name;
            dropZone.style.display = 'none';
            fileInfo.style.display = 'block';
            fileInfo.classList.add('fade-in');
        }
    }

    // Form Submit
    if (uploadForm) {
        uploadForm.addEventListener('submit', () => {
            loadingOverlay.style.display = 'flex';
        });
    }
});
