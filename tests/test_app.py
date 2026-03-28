def test_index(client):
    """Test that the index endpoint returns the API info."""
    response = client.get('/')
    assert response.status_code == 200
    
    data = response.get_json()
    assert "service" in data
    assert data["service"] == "Multi-Cloud Cost Optimization API"
    assert data["status"] == "running"

def test_404_not_found(client):
    """Test that a non-existent endpoint returns a 404 error."""
    response = client.get('/nonexistent-endpoint')
    assert response.status_code == 404
    
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Endpoint not found"
