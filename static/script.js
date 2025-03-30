document.addEventListener("DOMContentLoaded", () => {
    // Table row hover effects
    const rows = document.querySelectorAll("tbody tr");
    rows.forEach(row => {
        row.addEventListener("mouseenter", () => {
            row.style.transform = "scale(1.02)";
        });
        row.addEventListener("mouseleave", () => {
            row.style.transform = "scale(1)";
        });
    });
});