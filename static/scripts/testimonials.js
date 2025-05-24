const testimonials = [
    {
        name: "Dr. Sarah Thompson",
        role: "Scientist",
        review: "This platform has revolutionized the way we analyze satellite imagery for crop classification. The AI-driven models provide accurate results, helping researchers like me make data-driven decisions for sustainable farming.",
        image: "Img/person1.jpg"
    },
    {
        name: "John Doe",
        role: "Agricultural Engineer",
        review: "Using this AI-based crop mapping tool has significantly improved our yield predictions and planning strategies. The insights provided are incredibly valuable for modern farming.",
        image: "Img/person2.jpg"
    },
    {
        name: "Emily Carter",
        role: "Farm Owner",
        review: "Thanks to this platform, I can now make informed decisions about my crops. The satellite imagery and AI technology have made farming much more efficient.",
        image: "Img/person3.jpg"
    }
];

let currentIndex = 0;

function updateTestimonial(index) {
    document.getElementById("name").innerText = testimonials[index].name;
    document.getElementById("role").innerText = testimonials[index].role;
    document.getElementById("review").innerText = testimonials[index].review;
    document.getElementById("profile-pic").src = testimonials[index].image;
}

function nextTestimonial() {
    currentIndex = (currentIndex + 1) % testimonials.length;
    updateTestimonial(currentIndex);
}

function prevTestimonial() {
    currentIndex = (currentIndex - 1 + testimonials.length) % testimonials.length;
    updateTestimonial(currentIndex);
}

// Initialisation au premier témoignage
updateTestimonial(currentIndex);

