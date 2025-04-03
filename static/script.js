document.addEventListener("DOMContentLoaded", () => {
    // Hamburger Menu Toggle
    const hamburger = document.querySelector(".hamburger");
    const navMenu = document.querySelector(".nav-menu");

    hamburger.addEventListener("click", () => {
        navMenu.classList.toggle("active");
    });

    // Form Submission
    const form = document.getElementById("predict-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const loading = document.getElementById("loading-indicator");
        loading.style.display = "block";
        document.querySelectorAll(".error-message").forEach(el => el.textContent = "");

        const vehicleYear = form.vehicle_year.value;
        if (!vehicleYear || vehicleYear < 2017 || vehicleYear > 2025) {
            document.getElementById("vehicle-year-error").textContent = "Please select a year between 2017 and 2025.";
            loading.style.display = "none";
            return;
        }

        const formData = new FormData(form);
        try {
            const response = await fetch("/predict_json", {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Prediction failed: ${response.status} - ${errorText}`);
            }
            const data = await response.json();
            console.log("Prediction data:", data);

            updateCircularProgress("combined-ring", (data.combined_fuel / 15) * 100);
            updateCircularProgress("city-ring", (data.city_fuel / 20) * 100);
            updateCircularProgress("highway-ring", (data.highway_fuel / 15) * 100);
            updateCircularProgress("mpg-ring", (data.combined_mpg / 50) * 100);

            animateValue("combined-fuel", parseFloat(document.getElementById("combined-fuel").textContent), data.combined_fuel, 500);
            animateValue("city-fuel", parseFloat(document.getElementById("city-fuel").textContent), data.city_fuel, 500);
            animateValue("highway-fuel", parseFloat(document.getElementById("highway-fuel").textContent), data.highway_fuel, 500);
            animateValue("combined-mpg", parseFloat(document.getElementById("combined-mpg").textContent), data.combined_mpg, 500);
            animateValue("annual-fuel-cost", parseFloat(document.getElementById("annual-fuel-cost").textContent), data.annual_fuel_cost, 500);

            updateCircularProgress("co2-bar", (data.prediction / 450) * 100);
            document.getElementById("co2-value").textContent = `${data.prediction} g/km`;
            updateCircularProgress("co2-rating-bar", (data.co2_rating / 10) * 100);
            document.getElementById("co2-rating-value").textContent = data.co2_rating;
            updateCircularProgress("smog-bar", (data.smog_rating / 10) * 100);
            document.getElementById("smog-value").textContent = data.smog_rating;
            updateCircularProgress("eco-score-bar", (data.eco_score / 10) * 100);
            document.getElementById("eco-score-value").textContent = data.eco_score;

            const co2TaxEl = document.getElementById("co2-tax");
            co2TaxEl.textContent = data.co2_tax;
            console.log("CO2 Tax:", data.co2_tax);
            forceReflow(co2TaxEl);

            const fuelEffEl = document.getElementById("fuel-efficiency-score");
            fuelEffEl.textContent = data.fuel_efficiency_score;
            console.log("Fuel Efficiency Score:", data.fuel_efficiency_score);
            forceReflow(fuelEffEl);

            const costPerKmEl = document.getElementById("cost-per-passenger-km");
            costPerKmEl.textContent = data.cost_per_passenger_km;
            console.log("Cost per Passenger-Km:", data.cost_per_passenger_km);
            forceReflow(costPerKmEl);

        } catch (error) {
            console.error("Error in prediction:", error);
            alert("Failed to update predictions. Check the console for details.");
        } finally {
            loading.style.display = "none";
        }
    });

    // Reset Form
    document.getElementById("reset-form").addEventListener("click", () => {
        form.reset();
        form.vehicle_year.value = "2020";
        form.make.value = "Toyota";
        form.transmission.value = "Automated Manual";
        form.fuel_type.value = "X";

        updateCircularProgress("combined-ring", (9.0 / 15) * 100);
        updateCircularProgress("city-ring", (10.7 / 20) * 100);
        updateCircularProgress("highway-ring", (7.4 / 15) * 100);
        updateCircularProgress("mpg-ring", (31 / 50) * 100);

        animateValue("combined-fuel", parseFloat(document.getElementById("combined-fuel").textContent), 9.0, 500);
        animateValue("city-fuel", parseFloat(document.getElementById("city-fuel").textContent), 10.7, 500);
        animateValue("highway-fuel", parseFloat(document.getElementById("highway-fuel").textContent), 7.4, 500);
        animateValue("combined-mpg", parseFloat(document.getElementById("combined-mpg").textContent), 31, 500);
        animateValue("annual-fuel-cost", parseFloat(document.getElementById("annual-fuel-cost").textContent), 1962, 500);

        updateCircularProgress("co2-bar", (201.4 / 450) * 100);
        document.getElementById("co2-value").textContent = "201.4 g/km";
        updateCircularProgress("co2-rating-bar", (7 / 10) * 100);
        document.getElementById("co2-rating-value").textContent = "7";
        updateCircularProgress("smog-bar", (8 / 10) * 100);
        document.getElementById("smog-value").textContent = "8";
        updateCircularProgress("eco-score-bar", (7.5 / 10) * 100);
        document.getElementById("eco-score-value").textContent = "7.5";

        const co2TaxEl = document.getElementById("co2-tax");
        co2TaxEl.textContent = "N/A";
        forceReflow(co2TaxEl);

        const fuelEffEl = document.getElementById("fuel-efficiency-score");
        fuelEffEl.textContent = "N/A";
        forceReflow(fuelEffEl);

        const costPerKmEl = document.getElementById("cost-per-passenger-km");
        costPerKmEl.textContent = "N/A";
        forceReflow(costPerKmEl);
    });

    // QR Code Modal
    const qrButton = document.querySelector(".qr-button");
    const qrModal = document.getElementById("qr-modal");
    const close = document.querySelector(".close");

    qrButton.addEventListener("click", () => {
        qrModal.style.display = "block";
    });

    close.addEventListener("click", () => {
        qrModal.style.display = "none";
    });

    window.addEventListener("click", (e) => {
        if (e.target === qrModal) {
            qrModal.style.display = "none";
        }
    });

    // Animate Number Values
    function animateValue(id, start, end, duration) {
        const element = document.getElementById(id);
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = start + progress * (end - start);
            element.textContent = value.toFixed(id.includes("fuel") ? 1 : 0);
            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };
        requestAnimationFrame(step);
    }

    // Update Circular Progress
    function updateCircularProgress(id, percentage) {
        const circle = document.getElementById(id);
        if (!circle) {
            console.error(`Element with ID ${id} not found`);
            return;
        }
        console.log(`Updating ${id} with percentage: ${percentage}`);
        
        const clampedPercentage = Math.max(0, Math.min(100, percentage));
        const circumference = 314;
        const offset = circumference - (clampedPercentage / 100) * circumference;

        circle.style.strokeDashoffset = circumference;
        circle.offsetHeight;
        circle.style.strokeDashoffset = offset;
    }

    // Force Reflow Helper
    function forceReflow(element) {
        element.style.display = "none";
        element.offsetHeight;
        element.style.display = "";
    }

    // Initialize Circular Progress Rings
    updateCircularProgress("combined-ring", (9.0 / 15) * 100);
    updateCircularProgress("city-ring", (10.7 / 20) * 100);
    updateCircularProgress("highway-ring", (7.4 / 15) * 100);
    updateCircularProgress("mpg-ring", (31 / 50) * 100);
    updateCircularProgress("co2-bar", (201.4 / 450) * 100);
    updateCircularProgress("co2-rating-bar", (7 / 10) * 100);
    updateCircularProgress("smog-bar", (8 / 10) * 100);
    updateCircularProgress("eco-score-bar", (7.5 / 10) * 100);

    // Scroll Animations
    const sections = document.querySelectorAll(".about-section, .stats-section, .top-cars-section");
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
            }
        });
    }, { threshold: 0.2 });

    sections.forEach(section => observer.observe(section));

    // Fetch and Visualize Emission Insights
    async function fetchEmissionInsights() {
        try {
            const response = await fetch("/emission_insights");
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch emission insights: ${response.status} - ${errorText}`);
            }
            const data = await response.json();
            console.log("Emission insights data:", data);

            if (!data.fuel_emissions || !data.transmission_emissions || !data.engine_emissions || !data.top_5_combinations) {
                console.warn("Incomplete emission insights data");
                return;
            }

            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, ticks: { color: '#6E6E73' }, grid: { color: 'rgba(0, 0, 0, 0.05)' } },
                    x: { ticks: { color: '#6E6E73', maxRotation: 45, minRotation: 45 }, grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { backgroundColor: '#FF3B30', titleColor: '#FFFFFF', bodyColor: '#FFFFFF' }
                }
            };

            // Fuel Emissions Chart (Bar)
            new Chart(document.getElementById("fuelEmissionsChart"), {
                type: 'bar',
                data: {
                    labels: data.fuel_emissions.map(item => item.fuel_type),
                    datasets: [{
                        label: 'Avg CO₂ Emissions',
                        data: data.fuel_emissions.map(item => item.avg_co2),
                        backgroundColor: '#FF3B30',
                        borderColor: '#D32F2F',
                        borderWidth: 1
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        y: { title: { display: true, text: 'g/km', color: '#1D1D1F' } }
                    }
                }
            });

            // Transmission Emissions Chart (Bar)
            new Chart(document.getElementById("transmissionEmissionsChart"), {
                type: 'bar',
                data: {
                    labels: data.transmission_emissions.map(item => item.transmission),
                    datasets: [{
                        label: 'Avg CO₂ Emissions',
                        data: data.transmission_emissions.map(item => item.avg_co2),
                        backgroundColor: '#FF3B30',
                        borderColor: '#D32F2F',
                        borderWidth: 1
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        y: { title: { display: true, text: 'g/km', color: '#1D1D1F' } }
                    }
                }
            });

            // Engine Emissions Chart (Bar with error bars approximated)
            new Chart(document.getElementById("engineEmissionsChart"), {
                type: 'bar',
                data: {
                    labels: data.engine_emissions.map(item => item.engine_size),
                    datasets: [{
                        label: 'Mean CO₂ Emissions',
                        data: data.engine_emissions.map(item => item.mean_co2),
                        backgroundColor: '#FF3B30',
                        borderColor: '#D32F2F',
                        borderWidth: 1
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        y: { title: { display: true, text: 'g/km', color: '#1D1D1F' } }
                    }
                }
            });

            // Top 5 Combinations Chart (Bar)
            new Chart(document.getElementById("topCombinationsChart"), {
                type: 'bar',
                data: {
                    labels: data.top_5_combinations.map(item => `${item.engine_size}L, ${item.cylinders} cyl, ${item.transmission}, ${item.fuel_type}`),
                    datasets: [{
                        label: 'CO₂ Emissions',
                        data: data.top_5_combinations.map(item => item.co2_emissions),
                        backgroundColor: '#FF3B30',
                        borderColor: '#D32F2F',
                        borderWidth: 1
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        y: { title: { display: true, text: 'g/km', color: '#1D1D1F' } }
                    }
                }
            });
        } catch (error) {
            console.error("Error fetching emission insights:", error);
            alert("Failed to load emission insights charts. Check the console for details.");
        }
    }

    fetchEmissionInsights();
});
