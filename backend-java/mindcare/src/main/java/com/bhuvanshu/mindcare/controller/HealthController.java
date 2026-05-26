package com.bhuvanshu.mindcare.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@RestController
public class HealthController {

    private static final Logger logger = LoggerFactory.getLogger(HealthController.class);

    @Autowired
    private RestTemplate restTemplate;

    @Value("${ml.api.url}")
    private String mlApiBaseUrl;

    /**
     * Java backend health check.
     */
    @GetMapping("/api/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "up",
                "service", "mindcare-api",
                "version", "1.0.0"
        ));
    }

    /**
     * Proxies a health check to the Flask ML API (/health endpoint).
     * Returns the real ML service status so the frontend doesn't need
     * to call the ML API directly.
     */
    @GetMapping("/api/health/ml")
    public ResponseEntity<Map<String, Object>> mlHealth() {
        Map<String, Object> result = new HashMap<>();

        try {
            // Derive the ML health URL from the configured ml.api.url
            String mlHealthUrl = mlApiBaseUrl;
            if (mlHealthUrl.endsWith("/predict")) {
                mlHealthUrl = mlHealthUrl.replace("/predict", "/health");
            } else {
                mlHealthUrl = mlHealthUrl.endsWith("/")
                        ? mlHealthUrl + "health"
                        : mlHealthUrl + "/health";
            }

            ResponseEntity<Map> mlResponse = restTemplate.getForEntity(mlHealthUrl, Map.class);

            if (mlResponse.getStatusCode().is2xxSuccessful() && mlResponse.getBody() != null) {
                result.put("ml_status", "connected");
                result.put("ml_details", mlResponse.getBody());
            } else {
                result.put("ml_status", "error");
                result.put("ml_message", "ML API returned non-OK status");
            }
        } catch (Exception e) {
            logger.warn("ML health check failed: {}", e.getMessage());
            result.put("ml_status", "offline");
            result.put("ml_message", e.getMessage());
        }

        // Also report Java backend status
        result.put("backend_status", "up");
        result.put("service", "mindcare-api");

        return ResponseEntity.ok(result);
    }
}
