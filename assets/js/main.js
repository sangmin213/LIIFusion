(function () {
  const root = document.documentElement;
  const savedTheme = localStorage.getItem('liifusion-theme');
  if (savedTheme === 'dark') root.dataset.theme = 'dark';

  const themeToggle = document.querySelector('[data-theme-toggle]');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const nextTheme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      if (nextTheme === 'dark') root.dataset.theme = 'dark';
      else delete root.dataset.theme;
      localStorage.setItem('liifusion-theme', nextTheme);
    });
  }

  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('[data-nav-links]');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });
    navLinks.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const panel = document.querySelector('[data-lightbox-panel]');
  const panelImage = document.querySelector('[data-lightbox-image]');
  const closeButton = document.querySelector('[data-lightbox-close]');

  function closeLightbox() {
    if (!panel) return;
    panel.classList.remove('is-open');
    panel.setAttribute('aria-hidden', 'true');
    if (panelImage) panelImage.removeAttribute('src');
  }

  document.querySelectorAll('[data-lightbox]').forEach((image) => {
    image.addEventListener('click', () => {
      if (!panel || !panelImage) return;
      panelImage.src = image.src;
      panelImage.alt = image.alt || 'Expanded project figure';
      panel.classList.add('is-open');
      panel.setAttribute('aria-hidden', 'false');
    });
  });

  if (closeButton) closeButton.addEventListener('click', closeLightbox);
  if (panel) {
    panel.addEventListener('click', (event) => {
      if (event.target === panel) closeLightbox();
    });
  }
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeLightbox();
  });
})();
