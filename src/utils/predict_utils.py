def calculate_ppi(resolution: str, screen_size: float) -> float:
    """
    Convert resolution string and screen size to PPI.
    resolution: '1920x1080'
    screen_size: diagonal in inches
    """
    try:
        width, height = map(int, resolution.lower().split('x'))
        ppi = ((width**2 + height**2)**0.5) / screen_size
        return round(ppi, 2)
    except Exception:
        return 0  # fallback if input invalid