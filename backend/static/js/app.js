document.addEventListener('DOMContentLoaded', () => {
    // State management
    let state = {
        models: [],
        currentModel: 'smoke_test',
        results: [],
        filteredResults: [],
        activeFilter: 'all', // 'all', 'high', 'medium', 'all_risk'
        searchQuery: '',
        selectedResult: null,
        probMode: 'processed' // 'processed' or 'raw'
    };

    // UI Elements
    const modelSelect = document.getElementById('modelSelect');
    const probModeSelect = document.getElementById('probModeSelect');
    const csvDropZone = document.getElementById('csvDropZone');
    const csvInput = document.getElementById('csvInput');
    const csvStatus = document.getElementById('csvStatus');
    
    const imgDropZone = document.getElementById('imgDropZone');
    const imgInput = document.getElementById('imgInput');
    const imgStatus = document.getElementById('imgStatus');
    
    const predictBtn = document.getElementById('predictBtn');
    const consoleLog = document.getElementById('consoleLog');
    const emptyState = document.getElementById('emptyState');
    const resultsPanel = document.getElementById('resultsPanel');
    const resultsTableBody = document.getElementById('resultsTableBody');
    
    // Metrics
    const totalCount = document.getElementById('totalCount');
    const highRiskCount = document.getElementById('highRiskCount');
    const avgProbability = document.getElementById('avgProbability');
    
    // Actions
    const searchInput = document.getElementById('searchInput');
    const filterAll = document.getElementById('filterAll');
    const filterHigh = document.getElementById('filterHigh');
    const filterMedium = document.getElementById('filterMedium');
    const exportBtn = document.getElementById('exportBtn');
    
    // Modal
    const detailModal = document.getElementById('detailModal');
    const closeModal = document.getElementById('closeModal');

    // Utility: log to custom console
    function logToConsole(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        const time = new Date().toLocaleTimeString();
        line.innerText = `[${time}] ${message}`;
        consoleLog.appendChild(line);
        consoleLog.scrollTop = consoleLog.scrollHeight;
    }

    // Load available models on startup
    async function loadModels() {
        logToConsole('Đang truy vấn danh sách model khả dụng...', 'info');
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            state.models = data.models;
            
            modelSelect.innerHTML = '';
            if (state.models.length === 0) {
                const opt = document.createElement('option');
                opt.value = 'smoke_test';
                opt.innerText = 'smoke_test (Default Fallback)';
                modelSelect.appendChild(opt);
            } else {
                state.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.folder_name;
                    opt.innerText = `${m.preset_name} (${m.folder_name})`;
                    if (m.folder_name === 'smoke_test') opt.selected = true;
                    modelSelect.appendChild(opt);
                });
            }
            state.currentModel = modelSelect.value;
            logToConsole('Truy vấn danh sách model thành công.', 'success');
        } catch (e) {
            logToConsole(`Không thể tải danh sách model: ${e.message}`, 'error');
        }
    }

    // Set up model selector change handler
    modelSelect.addEventListener('change', (e) => {
        state.currentModel = e.target.value;
        logToConsole(`Đã chọn model: ${state.currentModel}`, 'info');
    });

    // Set up probability comparison mode change handler
    probModeSelect.addEventListener('change', (e) => {
        state.probMode = e.target.value;
        logToConsole(`Đã đổi chế độ so sánh kết quả sang: ${state.probMode === 'raw' ? 'Mô hình gốc' : 'Đã hiệu chỉnh (Patient-wise)'}`, 'info');
        applyFiltering();
        renderResults();
    });

    // File selection UI handling helpers
    function setupDragAndDrop(dropZone, fileInput, statusEl, allowedTypes, labelPrefix) {
        // Trigger file input click
        dropZone.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            handleFileSelect(e.target.files);
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFileSelect(e.dataTransfer.files);
        });

        function handleFileSelect(files) {
            if (files.length === 0) return;
            
            // For CSV we only allow 1 file
            if (allowedTypes.includes('csv')) {
                const file = files[0];
                if (file.name.endsWith('.csv')) {
                    statusEl.innerHTML = `
                        <span class="file-icon">📄</span>
                        <span class="file-name">${file.name} (${formatBytes(file.size)})</span>
                        <span class="clear-btn" id="clearCsv">✕</span>
                    `;
                    statusEl.style.display = 'flex';
                    logToConsole(`Đã chọn file CSV: ${file.name}`, 'info');
                    document.getElementById('clearCsv').addEventListener('click', (e) => {
                        e.stopPropagation();
                        fileInput.value = '';
                        statusEl.style.display = 'none';
                        logToConsole('Đã bỏ chọn file CSV', 'info');
                        updatePredictButtonState();
                    });
                } else {
                    logToConsole('Lỗi: File được chọn không phải là CSV!', 'error');
                }
            } else {
                // For images, could be ZIP, HDF5 or multiple image files
                if (files.length === 1 && files[0].name.endsWith('.zip')) {
                    const file = files[0];
                    statusEl.innerHTML = `
                        <span class="file-icon">📦</span>
                        <span class="file-name">${file.name} (${formatBytes(file.size)})</span>
                        <span class="clear-btn" id="clearImages">✕</span>
                    `;
                    statusEl.style.display = 'flex';
                    logToConsole(`Đã chọn tệp ZIP ảnh: ${file.name}`, 'info');
                } else if (files.length === 1 && (files[0].name.endsWith('.hdf5') || files[0].name.endsWith('.h5'))) {
                    const file = files[0];
                    statusEl.innerHTML = `
                        <span class="file-icon">🗄️</span>
                        <span class="file-name">${file.name} (${formatBytes(file.size)})</span>
                        <span class="clear-btn" id="clearImages">✕</span>
                    `;
                    statusEl.style.display = 'flex';
                    logToConsole(`Đã chọn tệp ảnh HDF5: ${file.name}`, 'info');
                } else {
                    statusEl.innerHTML = `
                        <span class="file-icon">🖼️</span>
                        <span class="file-name">Đã chọn ${files.length} ảnh</span>
                        <span class="clear-btn" id="clearImages">✕</span>
                    `;
                    statusEl.style.display = 'flex';
                    logToConsole(`Đã chọn ${files.length} ảnh riêng lẻ`, 'info');
                }
                
                document.getElementById('clearImages').addEventListener('click', (e) => {
                    e.stopPropagation();
                    fileInput.value = '';
                    statusEl.style.display = 'none';
                    logToConsole('Đã bỏ chọn ảnh', 'info');
                    updatePredictButtonState();
                });
            }
            updatePredictButtonState();
        }
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function updatePredictButtonState() {
        const hasCsv = csvInput.files.length > 0;
        predictBtn.disabled = !hasCsv;
    }

    setupDragAndDrop(csvDropZone, csvInput, csvStatus, ['csv'], 'CSV Metadata');
    setupDragAndDrop(imgDropZone, imgInput, imgStatus, ['zip', 'image'], 'Images');

    // Run prediction pipeline
    predictBtn.addEventListener('click', async () => {
        if (csvInput.files.length === 0) return;
        
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="spinner"></span> Đang xử lý...';
        
        logToConsole('Khởi động Pipeline phân tích bệnh học...', 'info');
        logToConsole(`Đang upload các file dữ liệu lên server...`, 'info');
        
        const formData = new FormData();
        formData.append('csv_file', csvInput.files[0]);
        formData.append('model_folder', state.currentModel);
        
        // Add zip, hdf5 or images file
        const imagesInputFiles = imgInput.files;
        if (imagesInputFiles.length > 0) {
            if (imagesInputFiles.length === 1 && imagesInputFiles[0].name.endsWith('.zip')) {
                formData.append('zip_file', imagesInputFiles[0]);
            } else if (imagesInputFiles.length === 1 && (imagesInputFiles[0].name.endsWith('.hdf5') || imagesInputFiles[0].name.endsWith('.h5'))) {
                formData.append('hdf5_file', imagesInputFiles[0]);
            } else {
                for (let i = 0; i < imagesInputFiles.length; i++) {
                    formData.append('image_files', imagesInputFiles[i]);
                }
            }
        }
        
        try {
            logToConsole('Server đang giải nén ảnh và thực hiện Feature Engineering...', 'info');
            logToConsole('Đang chạy suy luận với tổ hợp (ensemble) các model checkpoint...', 'info');
            
            const start = performance.now();
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            const duration = ((performance.now() - start) / 1000).toFixed(2);
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Lỗi server không xác định.');
            }
            
            const data = await res.json();
            logToConsole(`Pipeline chạy hoàn thành trong ${duration}s!`, 'success');
            
            state.results = data.results;
            applyFiltering();
            renderResults();
            
        } catch (e) {
            logToConsole(`Lỗi Pipeline: ${e.message}`, 'error');
            alert(`Đã xảy ra lỗi khi chạy pipeline: ${e.message}`);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '⚡ Chạy Pipeline Xử Lý';
        }
    });

    // Filtering, Search, and Sorting Logic
    function applyFiltering() {
        let items = [...state.results];
        
        // Apply search query
        if (state.searchQuery) {
            const query = state.searchQuery.toLowerCase();
            items = items.filter(r => 
                r.isic_id.toLowerCase().includes(query) || 
                r.patient_id.toLowerCase().includes(query) ||
                (r.anatom_site_general && r.anatom_site_general.toLowerCase().includes(query))
            );
        }
        
        const probKey = state.probMode === 'raw' ? 'probability_raw' : 'probability_processed';
        
        // Apply risk filter
        if (state.activeFilter === 'high') {
            items = items.filter(r => r[probKey] >= 0.5);
        } else if (state.activeFilter === 'medium') {
            items = items.filter(r => r[probKey] >= 0.15 && r[probKey] < 0.5);
        }
        
        // Sort by risk descending
        items.sort((a, b) => b[probKey] - a[probKey]);
        
        state.filteredResults = items;
    }

    function renderResults() {
        if (state.results.length === 0) {
            emptyState.style.display = 'flex';
            resultsPanel.style.display = 'none';
            return;
        }
        
        emptyState.style.display = 'none';
        resultsPanel.style.display = 'block';
        
        // Render Metrics
        totalCount.innerText = state.results.length;
        
        const probKey = state.probMode === 'raw' ? 'probability_raw' : 'probability_processed';
        
        const highRisk = state.results.filter(r => r[probKey] >= 0.5).length;
        highRiskCount.innerText = highRisk;
        if (highRisk > 0) {
            highRiskCount.className = 'metric-value high-risk';
        } else {
            highRiskCount.className = 'metric-value';
        }
        
        const avg = state.results.reduce((sum, r) => sum + r[probKey], 0) / state.results.length;
        avgProbability.innerText = `${(avg * 100).toFixed(1)}%`;
        
        // Render Table
        resultsTableBody.innerHTML = '';
        if (state.filteredResults.length === 0) {
            resultsTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">Không tìm thấy kết quả phù hợp</td></tr>`;
            return;
        }
        
        state.filteredResults.forEach(r => {
            const tr = document.createElement('tr');
            tr.dataset.id = r.isic_id;
            
            // Risk Badge
            let badgeClass = 'success';
            let badgeText = 'Thấp';
            const probVal = r[probKey];
            if (probVal >= 0.5) {
                badgeClass = 'danger';
                badgeText = 'Cao';
            } else if (probVal >= 0.15) {
                badgeClass = 'warning';
                badgeText = 'Trung bình';
            }
            
            const rawHighlight = state.probMode === 'raw' ? 'font-weight: 700; color: var(--primary-color);' : '';
            const processedHighlight = state.probMode === 'processed' ? 'font-weight: 700; color: var(--primary-color);' : '';
            
            tr.innerHTML = `
                <td><img src="${r.image_url}" class="cell-thumbnail" alt="lesion" /></td>
                <td><span class="cell-id">${r.isic_id}</span></td>
                <td><span class="cell-patient">${r.patient_id}</span></td>
                <td>
                    <div class="cell-details-summary">
                        ${r.age_approx !== null ? `${r.age_approx}t` : 'N/A'} • 
                        ${r.sex || 'N/A'} • 
                        ${r.anatom_site_general || 'N/A'}
                    </div>
                </td>
                <td><span class="badge ${badgeClass}">${badgeText}</span></td>
                <td><span class="cell-prob" style="${rawHighlight}">${(r.probability_raw * 100).toFixed(2)}%</span></td>
                <td><span class="cell-prob" style="${processedHighlight}">${(r.probability_processed * 100).toFixed(2)}%</span></td>
            `;
            
            tr.addEventListener('click', () => openDetailModal(r.isic_id));
            resultsTableBody.appendChild(tr);
        });
    }

    // Search and Filter Handlers
    searchInput.addEventListener('input', (e) => {
        state.searchQuery = e.target.value;
        applyFiltering();
        renderResults();
    });

    const filterButtons = [
        { btn: filterAll, filter: 'all' },
        { btn: filterHigh, filter: 'high' },
        { btn: filterMedium, filter: 'medium' }
    ];

    filterButtons.forEach(cfg => {
        cfg.btn.addEventListener('click', () => {
            filterButtons.forEach(c => c.btn.classList.remove('active'));
            cfg.btn.classList.add('active');
            state.activeFilter = cfg.filter;
            applyFiltering();
            renderResults();
        });
    });

    // Details Modal
    function openDetailModal(isicId) {
        const item = state.results.find(r => r.isic_id === isicId);
        if (!item) return;
        
        state.selectedResult = item;
        
        // Set Modal Content
        document.getElementById('modalImage').src = item.image_url;
        
        const camContainer = document.getElementById('modalCamContainer');
        const camImage = document.getElementById('modalCamImage');
        if (item.cam_image_url) {
            camImage.src = item.cam_image_url;
            camContainer.style.display = 'block';
        } else {
            camContainer.style.display = 'none';
            camImage.src = '';
        }
        
        document.getElementById('modalIsicId').innerText = item.isic_id;
        document.getElementById('modalPatientId').innerText = item.patient_id;
        document.getElementById('modalAge').innerText = item.age_approx !== null ? `${item.age_approx} tuổi` : 'Không xác định';
        document.getElementById('modalSex').innerText = item.sex || 'Không xác định';
        document.getElementById('modalSite').innerText = item.anatom_site_general || 'Không xác định';
        document.getElementById('modalSize').innerText = item.clin_size_long_diam_mm !== null ? `${item.clin_size_long_diam_mm} mm` : 'Không xác định';
        
        // Display Risk Badge in Modal
        const riskLabel = document.getElementById('modalRiskLabel');
        let badgeClass = 'success';
        let badgeText = 'Nguy cơ ác tính: THẤP';
        
        const probKey = state.probMode === 'raw' ? 'probability_raw' : 'probability_processed';
        const probVal = item[probKey];
        if (probVal >= 0.5) {
            badgeClass = 'danger';
            badgeText = 'Nguy cơ ác tính: CAO';
        } else if (probVal >= 0.15) {
            badgeClass = 'warning';
            badgeText = 'Nguy cơ ác tính: TRUNG BÌNH';
        }
        riskLabel.className = `badge ${badgeClass}`;
        riskLabel.innerText = badgeText;
        
        const rawProbEl = document.getElementById('modalRawProb');
        const processedProbEl = document.getElementById('modalProcessedProb');
        if (state.probMode === 'raw') {
            rawProbEl.style.color = 'var(--primary-color)';
            rawProbEl.style.fontSize = '1.1rem';
            processedProbEl.style.color = 'var(--text-color)';
            processedProbEl.style.fontSize = '1rem';
        } else {
            rawProbEl.style.color = 'var(--text-color)';
            rawProbEl.style.fontSize = '1rem';
            processedProbEl.style.color = 'var(--primary-color)';
            processedProbEl.style.fontSize = '1.1rem';
        }
        
        rawProbEl.innerText = `${(item.probability_raw * 100).toFixed(4)}%`;
        processedProbEl.innerText = `${(item.probability_processed * 100).toFixed(4)}%`;
        
        detailModal.classList.add('active');
    }

    closeModal.addEventListener('click', () => {
        detailModal.classList.remove('active');
    });

    detailModal.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            detailModal.classList.remove('active');
        }
    });

    // Export Results CSV
    exportBtn.addEventListener('click', () => {
        if (state.results.length === 0) return;
        
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "isic_id,patient_id,age_approx,sex,anatom_site_general,clin_size_long_diam_mm,probability_raw,probability_processed,risk_level\n";
        
        state.results.forEach(r => {
            let risk = 'Low';
            if (r.probability_processed >= 0.5) risk = 'High';
            else if (r.probability_processed >= 0.15) risk = 'Medium';
            
            const line = [
                r.isic_id,
                r.patient_id,
                r.age_approx !== null ? r.age_approx : '',
                r.sex || '',
                `"${r.anatom_site_general || ''}"`,
                r.clin_size_long_diam_mm !== null ? r.clin_size_long_diam_mm : '',
                r.probability_raw,
                r.probability_processed,
                risk
            ].join(",");
            csvContent += line + "\n";
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `skin_lesion_analysis_${state.currentModel}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Init
    loadModels();
});
