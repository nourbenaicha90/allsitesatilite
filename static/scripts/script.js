document.addEventListener("DOMContentLoaded", () => {
    console.log("Page Loaded");

    // Ajoute une interaction pour le bouton "Get Started"
    document.getElementById("get-started-btn").addEventListener("click", function() {
        alert("🚀 Welcome to AgriSat! Let's explore AI-powered crop mapping.");
    });
});
// 🌟 Ajout d’une animation de survol sur le logo du footer
document.addEventListener("DOMContentLoaded", function () {
    let logo = document.querySelector(".footer-logo");

    logo.addEventListener("mouseenter", function () {
        this.style.transform = "scale(1.1)";
        this.style.transition = "0.3s ease-in-out";
    });

    logo.addEventListener("mouseleave", function () {
        this.style.transform = "scale(1)";
    });
});
