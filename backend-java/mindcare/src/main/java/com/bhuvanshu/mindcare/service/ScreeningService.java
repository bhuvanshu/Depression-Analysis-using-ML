package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.ScreeningRequest;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class ScreeningService {

    @Autowired
    private RestTemplate restTemplate;

    private final String ML_API =
            "http://localhost:5000/predict";

    public String submitScreening(
            ScreeningRequest request) {

        HttpHeaders headers = new HttpHeaders();

        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<ScreeningRequest> entity =
                new HttpEntity<>(request, headers);

        ResponseEntity<String> response =
                restTemplate.postForEntity(
                        ML_API,
                        entity,
                        String.class
                );

        return response.getBody();
    }
}