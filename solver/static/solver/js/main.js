document.addEventListener('DOMContentLoaded', () => {
    // Form submission handling for loading states
    const pinnForm = document.getElementById('pinn-form');
    const surrogateForm = document.getElementById('surrogate-form');
    
    function handleFormSubmit(formElement) {
        if (!formElement) return;
        
        formElement.addEventListener('submit', (e) => {
            const submitBtn = document.getElementById('submit-btn');
            const loading = document.getElementById('loading');
            
            if (submitBtn && loading) {
                submitBtn.classList.add('hidden');
                loading.classList.remove('hidden');
            }
        });
    }
    
    handleFormSubmit(pinnForm);
    handleFormSubmit(surrogateForm);
});
