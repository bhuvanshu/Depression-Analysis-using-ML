package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.ScreeningRequest;
import com.bhuvanshu.mindcare.dto.ScreeningResultResponse;
import com.bhuvanshu.mindcare.service.ScreeningService;

import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/screening")
public class ScreeningController {

    private static final Logger logger = LoggerFactory.getLogger(ScreeningController.class);

    @Autowired
    private ScreeningService screeningService;

    @PostMapping("/submit")
    public ResponseEntity<?> submitScreening(
            @Valid @RequestBody ScreeningRequest request) {

        try {
            logger.info("POST /screening/submit | enrollmentId={}", request.getEnrollmentId());
            ScreeningResultResponse response = screeningService.submitScreening(request);
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            logger.error("Error in screening submission for enrollmentId={}: {}",
                    request.getEnrollmentId(), e.getMessage(), e);
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }
}