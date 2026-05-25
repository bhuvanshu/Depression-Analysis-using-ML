package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.ScreeningRequest;
import com.bhuvanshu.mindcare.dto.ScreeningResultResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResult;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.ScreeningResponseRepository;
import com.bhuvanshu.mindcare.repository.ScreeningResultRepository;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;

@Service
public class ScreeningService {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private StudentRepository studentRepository;

    @Autowired
    private ScreeningResponseRepository screeningResponseRepository;

    @Autowired
    private ScreeningResultRepository screeningResultRepository;

    @Value("${ml.api.url}")
    private String ML_API_BASE_URL;

    public ScreeningResultResponse submitScreening(ScreeningRequest request) {

        // FETCH STUDENT
        Student student = studentRepository
                .findByEnrollmentId(request.getEnrollmentId())
                .orElseThrow(() -> new RuntimeException("Student not found with ID: " + request.getEnrollmentId()));

        // SAVE QUESTIONNAIRE RESPONSE
        ScreeningResponse responseEntity = new ScreeningResponse();
        responseEntity.setStudent(student);

        responseEntity.setAcademicPressure(
                request.getAcademicPressure());

        responseEntity.setFinancialStress(
                request.getFinancialStress());

        responseEntity.setStudySatisfaction(
                request.getStudySatisfaction());

        responseEntity.setWorkStudyHours(
                request.getWorkStudyHours());

        responseEntity.setCgpa(
                request.getCgpa());

        responseEntity.setSleepDuration(
                request.getSleepDuration());

        responseEntity.setFamilyHistory(
                request.getFamilyHistory());

        responseEntity.setSuicidalThoughts(
                request.getSuicidalThoughts());

        screeningResponseRepository.save(
                responseEntity);

        // CALL ML API
        String predictUrl = ML_API_BASE_URL;
        if (predictUrl == null || predictUrl.trim().isEmpty()) {
            throw new RuntimeException("ML_API_URL environment variable is not configured");
        }

        // Ensure URL ends with /predict
        if (!predictUrl.endsWith("/predict")) {
            predictUrl = predictUrl.endsWith("/") ? predictUrl + "predict" : predictUrl + "/predict";
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<ScreeningRequest> entity = new HttpEntity<>(request, headers);

        ScreeningResultResponse prediction;
        try {
            ResponseEntity<ScreeningResultResponse> mlResponse = restTemplate.postForEntity(
                    predictUrl,
                    entity,
                    ScreeningResultResponse.class);
            prediction = mlResponse.getBody();
        } catch (RestClientException e) {
            throw new RuntimeException("Failed to reach ML API at " + predictUrl + ": " + e.getMessage(), e);
        }

        if (prediction == null) {
            throw new RuntimeException("ML prediction response is empty");
        }

        // SAVE PREDICTION RESULT

        ScreeningResult result = new ScreeningResult();
        result.setScreeningResponse(responseEntity);
        result.setPredictedAt(LocalDateTime.now());

        // Null safety for database constraints
        String riskLevel = prediction.getRiskLevel();
        result.setRiskLevel(riskLevel != null ? riskLevel : "Unknown");

        if (prediction.getSeverityInterpretation() != null && prediction.getSeverityInterpretation().getLevel() != null) {
            result.setSeverityLevel(prediction.getSeverityInterpretation().getLevel());
        } else {
            result.setSeverityLevel("Unknown");
        }

        String recommendation = prediction.getRecommendedAction();
        result.setRecommendation(recommendation != null ? recommendation : "No recommendation provided");

        // SAFE PROBABILITY HANDLING

        if (prediction.getProbability() != null
                && prediction.getProbability()
                        .getDepressed() != null) {

            result.setProbabilityScore(
                    prediction.getProbability()
                            .getDepressed());

        } else {

            result.setProbabilityScore(0.0);
        }

        screeningResultRepository.save(result);

        return prediction;
    }
}