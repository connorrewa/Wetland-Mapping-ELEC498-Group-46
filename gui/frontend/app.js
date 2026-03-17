// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:5000',
    WETLAND_CLASSES: {
        0: { name: 'Background', color: '#1a1a2e' },
        1: { name: 'Fen (Graminoid)', color: '#7289da' },
        2: { name: 'Fen (Woody)', color: '#43b581' },
        3: { name: 'Marsh', color: '#16c79a' },
        4: { name: 'Shallow Open Water', color: '#ee5a6f' },
        5: { name: 'Swamp', color: '#faa61a' }
    },
    MAP_CENTER: [51.0447, -114.0719],
    MAP_ZOOM: 10,
    GEOTIFF_BASE_URL: 'http://localhost:5000/api/geotiff'
};


// State
let classificationResults = null;
let map = null;
let chart = null;
let geotiffLayer = null; // Direct reference — used for reliable removal
const arrayBufferCache = {};


// DOM Elements
const loadingOverlay = document.getElementById('loadingOverlay');
const progressFill = document.getElementById('progressFill');
const resultsSection = document.getElementById('resultsSection');
const mapPlaceholder = document.getElementById('mapPlaceholder');
const mapElement = document.getElementById('map');
const tifSelector = document.getElementById('tifSelector');
const loadingText = document.getElementById('loadingText');


// Initialize Application
async function init() {
    setupEventListeners();
    initializeMap();
    await loadFileList();
    console.log('🌿 Wetland Mapping Application Initialized');
}


// Event Listeners
function setupEventListeners() {
    tifSelector.addEventListener('change', (e) => {
        const selectedIndex = e.target.selectedIndex;
        const selectedOption = e.target.options[selectedIndex];

        // Strictly prevent selection of disabled options
        if (selectedOption && selectedOption.disabled) {
            for (let i = 0; i < e.target.options.length; i++) {
                if (!e.target.options[i].disabled) {
                    e.target.selectedIndex = i;
                    break;
                }
            }
            return;
        }

        if (e.target.value) {
            fetchResults(e.target.value);
        }
    });
}


// Fetch list of files and populate selector
async function loadFileList() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/files`);
        if (!response.ok) throw new Error('Could not load file list');

        const files = await response.json();

        tifSelector.innerHTML = '';
        if (files.length === 0) {
            tifSelector.innerHTML = '<option value="">No files found</option>';
            return;
        }

        files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;

            if (file.includes('RF')) {
                option.selected = true;
            } else {
                option.disabled = true;
                option.textContent = `${file} (Preloading in background...)`;
                preloadGeoTIFF(file, option);
            }

            tifSelector.appendChild(option);
        });

        if (tifSelector.value) {
            fetchResults(tifSelector.value);
        }
    } catch (error) {
        console.error('Error fetching file list:', error);
        tifSelector.innerHTML = '<option value="">Error loading files</option>';
    }
}


// Preload GeoTIFF in background
async function preloadGeoTIFF(filename, optionElement) {
    try {
        console.log(`📥 Preloading ${filename} in background...`);
        const url = `${CONFIG.GEOTIFF_BASE_URL}?file=${encodeURIComponent(filename)}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Preload fetch failed: ${response.status}`);

        const arrayBuffer = await response.arrayBuffer();
        arrayBufferCache[filename] = arrayBuffer;
        console.log(`✅ Preloaded and cached ${filename}`);

        optionElement.disabled = false;
        optionElement.textContent = filename;
    } catch (err) {
        console.error(`❌ Failed to preload ${filename}:`, err);
        optionElement.textContent = `${filename} (Failed to load)`;
    }
}


// Initialize Leaflet Map
function initializeMap() {
    map = L.map('map', {
        center: CONFIG.MAP_CENTER,
        zoom: CONFIG.MAP_ZOOM,
        zoomControl: true
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);

    const marker = L.marker(CONFIG.MAP_CENTER).addTo(map);
    marker.bindPopup('<strong>Bow River Basin</strong><br>Alberta, Canada');
}


// Fetch Classification Results from Backend
async function fetchResults(filename) {
    if (!filename) return;

    try {
        showLoading(true, `Loading data for ${filename}...`);
        updateProgress(10);

        const startTime = performance.now();

        updateProgress(30);

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/results?file=${encodeURIComponent(filename)}`);

        updateProgress(70);

        if (!response.ok) {
            throw new Error('Could not load results. Please check if the backend server is running.');
        }

        const results = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);

        updateProgress(100);

        classificationResults = {
            ...results,
            processingTime
        };

        await displayResults(classificationResults);

    } catch (error) {
        console.error('Error fetching results:', error);
        showNotification(error.message, 'error');
    } finally {
        showLoading(false);
    }
}


// Display Results
async function displayResults(results) {
    resultsSection.classList.add('active');

    document.getElementById('statTotal').textContent = formatNumber(results.total_samples || 0);
    document.getElementById('statAccuracy').textContent = results.confidence ? `${(results.confidence * 100).toFixed(1)}%` : 'N/A';
    document.getElementById('statTime').textContent = `${results.processingTime}s`;

    if (results.class_distribution) {
        Object.keys(results.class_distribution).forEach(classId => {
            const count = results.class_distribution[classId];
            const percentage = ((count / results.total_samples) * 100).toFixed(1);
            const legendValue = document.querySelector(`.legend-value[data-class="${classId}"]`);
            if (legendValue) {
                legendValue.textContent = `${percentage}%`;
            }
        });
    }

    updateChart(results.class_distribution);
    await showMapVisualization(results, document.getElementById('tifSelector').value);

    console.log('✅ Results displayed');
}


// Update Chart
function updateChart(distribution) {
    const ctx = document.getElementById('distributionChart');

    if (chart) {
        chart.destroy();
    }

    const labels = Object.keys(CONFIG.WETLAND_CLASSES).map(id => CONFIG.WETLAND_CLASSES[id].name);
    const data = Object.keys(CONFIG.WETLAND_CLASSES).map(id => distribution[id] || 0);
    const colors = Object.keys(CONFIG.WETLAND_CLASSES).map(id => CONFIG.WETLAND_CLASSES[id].color);

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sample Count',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c + '80'),
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 46, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#b8b9cf',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return `Count: ${formatNumber(context.parsed.y)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#7885a3',
                        callback: function (value) { return formatNumber(value); }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#7885a3' }
                }
            }
        }
    });
}


// Hex colour string → [r, g, b] array (values 0-255)
function hexToRgb(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return [r, g, b];
}


// Build a lookup: class id (0-5) → [r, g, b]
const CLASS_COLORS = Object.fromEntries(
    Object.entries(CONFIG.WETLAND_CLASSES).map(([id, cls]) => [Number(id), hexToRgb(cls.color)])
);


// Show Map Visualization — fetches GeoTIFF and renders it on the Leaflet map
async function showMapVisualization(results, filename) {
    if (!filename) return;

    // FIX 1: Always remove the old layer FIRST, before any early returns
    if (geotiffLayer) {
        map.removeLayer(geotiffLayer);
        geotiffLayer = null;
    }

    // FIX 2: Unhide map and immediately invalidate size (no fragile setTimeout)
    mapPlaceholder.classList.add('hidden');
    mapElement.classList.remove('hidden');
    map.invalidateSize();

    if (!results.geotiff_ready) {
        // FIX 3: Notify the user instead of silently failing
        console.warn('⚠️ GeoTIFF not ready — skipping map overlay');
        showNotification('Map overlay unavailable for this file.', 'info');
        return;
    }

    try {
        let arrayBuffer;
        if (arrayBufferCache[filename]) {
            console.log(`🗺️ Using preloaded GeoTIFF for ${filename}...`);
            arrayBuffer = arrayBufferCache[filename].slice(0);
        } else {
            console.log(`🗺️ Fetching GeoTIFF (${filename}) from backend...`);
            const url = `${CONFIG.GEOTIFF_BASE_URL}?file=${encodeURIComponent(filename)}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`GeoTIFF fetch failed: ${response.status}`);

            const fetchedBuffer = await response.arrayBuffer();
            arrayBufferCache[filename] = fetchedBuffer;
            arrayBuffer = fetchedBuffer.slice(0);
        }

        const georaster = await parseGeoraster(arrayBuffer);

        // FIX 4: Add directly to map — GeoRasterLayer does not reliably
        // clean up when removed via L.layerGroup().clearLayers()
        geotiffLayer = new GeoRasterLayer({
            georaster,
            opacity: 0.75,
            pixelValuesToColorFn: (values) => {
                const classId = values[0];
                if (classId === 255 || classId === undefined) return null;
                const rgb = CLASS_COLORS[classId];
                if (!rgb) return null;
                return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            },
            resolution: 256,
        });

        geotiffLayer.addTo(map);
        map.fitBounds(geotiffLayer.getBounds());

        console.log('✅ GeoTIFF overlay rendered on map');

    } catch (err) {
        console.error('❌ GeoTIFF overlay failed:', err);
        showNotification('Could not render map overlay: ' + err.message, 'error');
    }
}


// Utility Functions
function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function showLoading(show, message = 'Loading data...') {
    if (show) {
        if (loadingText) loadingText.textContent = message;
        loadingOverlay.classList.remove('hidden');
        if (tifSelector) tifSelector.disabled = true;
        updateProgress(0);
    } else {
        setTimeout(() => {
            loadingOverlay.classList.add('hidden');
            if (tifSelector) tifSelector.disabled = false;
        }, 500);
    }
}

function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
}

function showNotification(message, type = 'info') {
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    console.log(`${icon} ${message}`);
    alert(message);
}


// Initialize on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
