package com.bhuvanshu.mindcare.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "students")
public class Student {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long studentId;

    @Column(nullable = false, unique = true)
    private String enrollmentId;

    @Column(nullable = false)
    private String name;

    private Integer age;

    private String gender;

    private String department;

    private String degreeGroup;

    private LocalDateTime createdAt = LocalDateTime.now();

    @ManyToOne
    @JoinColumn(name = "college_id")
    private College college;

    // Getters and Setters
}