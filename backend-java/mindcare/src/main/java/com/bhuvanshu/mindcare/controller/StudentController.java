package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.BulkAddResponse;
import com.bhuvanshu.mindcare.dto.StudentAddRequest;
import com.bhuvanshu.mindcare.dto.StudentResponse;
import com.bhuvanshu.mindcare.dto.StudentVerifyRequest;
import com.bhuvanshu.mindcare.service.StudentService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/student")
@CrossOrigin("*")
public class StudentController {

    @Autowired
    private StudentService studentService;

    @PostMapping("/verify")
    public ResponseEntity<?> verifyStudent(
            @RequestBody StudentVerifyRequest request) {

        StudentResponse response =
                studentService.verifyStudent(request);

        if (response == null) {
            return ResponseEntity
                    .badRequest()
                    .body("Student not found");
        }

        return ResponseEntity.ok(response);
    }

    @PostMapping("/add")
    public ResponseEntity<?> addStudent(
            @RequestBody StudentAddRequest request,
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {
        try {
            StudentResponse response = studentService.addStudent(request, collegeName);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("An error occurred while adding the student");
        }
    }

    @PostMapping("/bulk")
    public ResponseEntity<?> bulkAddStudents(
            @RequestBody List<StudentAddRequest> requests,
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {
        try {
            BulkAddResponse response = studentService.bulkAddStudents(requests, collegeName);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("An error occurred during bulk upload");
        }
    }
}