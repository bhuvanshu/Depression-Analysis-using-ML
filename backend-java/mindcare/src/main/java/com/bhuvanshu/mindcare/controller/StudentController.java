package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.StudentResponse;
import com.bhuvanshu.mindcare.dto.StudentVerifyRequest;
import com.bhuvanshu.mindcare.service.StudentService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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
}