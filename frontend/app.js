const API_URL = 'http://localhost:5000'; // Flask API directly for now, or Spring Boot proxy

function nextStep(step) {
    document.querySelectorAll('.form-step').forEach(el => el.classList.remove('active'));
    document.getElementById(`step${step}`).classList.add('active');
}

document.getElementById('screeningForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = isNaN(value) ? value : parseFloat(value);
    });

    // Special handling for radio/checkboxes if needed
    const suicidal = document.querySelector('input[name="Suicidal Thoughts"]:checked').value;
    const family = document.querySelector('input[name="Family History"]:checked').value;
    data["Suicidal Thoughts"] = parseInt(suicidal);
    data["Family History"] = parseInt(family);

    // Map other missing features if needed (using defaults from config.py logic)
    // Gender mapping is already in values (1/0)

    const submitBtn = e.target.querySelector('.btn-submit');
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    submitBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.status === 'success') {
            showResult(result);
        } else {
            alert("Error analyzing results: " + result.message);
        }
    } catch (error) {
        console.error("API Error:", error);
        alert("Could not connect to the AI service. Please ensure the backend is running.");
    } finally {
        submitBtn.innerHTML = 'Analyze My Results <i class="fas fa-brain"></i>';
        submitBtn.disabled = false;
    }
});

function showResult(result) {
    document.querySelector('.form-container').classList.add('hidden');
    document.getElementById('resultContainer').classList.remove('hidden');

    const badge = document.getElementById('predictionBadge');
    badge.innerText = result.prediction_label;
    badge.className = `badge ${result.prediction_label}`;

    // Set probability
    const prob = result.probability[result.prediction] * 100;
    document.getElementById('probBar').style.width = `${prob}%`;

    const suggestionList = document.getElementById('suggestionList');
    suggestionList.innerHTML = '';

    const advice = {
        'Healthy': [
            "Good job! Your mental health screening indicates low risk.",
            "Continue practicing mindfulness and regular physical activity.",
            "Stay connected with friends and family."
        ],
        'Depressed': [
            "We recommend reaching out to a professional counselor.",
            "Try to maintain a fixed routine for sleep and meals.",
            "Consider talking to a trusted mentor or friend about your feelings.",
            "Remember, seeking help is a sign of strength."
        ]
    };

    advice[result.prediction_label].forEach(text => {
        const li = document.createElement('li');
        li.innerText = text;
        suggestionList.appendChild(li);
    });
}
