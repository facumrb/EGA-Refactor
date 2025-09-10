document.addEventListener("keydown", function(event) {
    // Extrae el número de página actual desde el nombre del archivo
    const match = window.location.pathname.match(/slide(\d+)\.html/);
    if (!match) return; // Si no coincide, no hace nada

    let currentPage = parseInt(match[1]);
    totalPages = 18;

    if (event.key === "ArrowRight" && currentPage < totalPages) {
      // Navega a la siguiente página
      window.location.href = `slide${currentPage + 1}.html`;
    } else if (event.key === "ArrowLeft" && currentPage > 1) {
      // Navega a la página anterior (si no es la primera)
      window.location.href = `slide${currentPage - 1}.html`;
    }
  });
