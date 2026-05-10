package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BulkAddResponse {
    private int totalUploaded;
    private int skippedDuplicates;
    private String message;
}
