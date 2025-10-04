# fluxgraph/security/pii_detector.py
"""
PII (Personally Identifiable Information) Detection for FluxGraph.
Detects and redacts sensitive information in agent inputs/outputs.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_RECORD = "medical_record"
    BANK_ACCOUNT = "bank_account"


class PIIPattern:
    """Regex patterns for PII detection."""
    
    PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
        PIIType.SSN: r'\b\d{3}-?\d{2}-?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        PIIType.PASSPORT: r'\b[A-Z]{1,2}[0-9]{6,9}\b',
        PIIType.DATE_OF_BIRTH: r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
        PIIType.MEDICAL_RECORD: r'\b(?:MRN|Medical Record):?\s*[A-Z0-9]{6,12}\b',
        PIIType.BANK_ACCOUNT: r'\b(?:Account|Acct):?\s*[0-9]{8,17}\b',
    }


class PIIDetection:
    """Represents detected PII in text."""
    
    def __init__(
        self,
        pii_type: PIIType,
        value: str,
        start_pos: int,
        end_pos: int,
        confidence: float = 1.0
    ):
        self.pii_type = pii_type
        self.value = value
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pii_type.value,
            "value": self.value[:4] + "***",  # Redact in logs
            "position": f"{self.start_pos}-{self.end_pos}",
            "confidence": self.confidence
        }


class PIIDetector:
    """
    Detects and redacts PII from text using regex patterns.
    Supports GDPR, HIPAA, and other compliance requirements.
    """
    
    def __init__(
        self,
        enabled_types: Optional[List[PIIType]] = None,
        redaction_char: str = "X"
    ):
        self.enabled_types = enabled_types or list(PIIType)
        self.redaction_char = redaction_char
        self.detection_count: Dict[PIIType, int] = {pii_type: 0 for pii_type in PIIType}
        logger.info(f"PIIDetector initialized for types: {[t.value for t in self.enabled_types]}")
    
    def detect(self, text: str) -> List[PIIDetection]:
        """
        Detect all PII in the given text.
        
        Args:
            text: Text to scan for PII
        
        Returns:
            List of detected PII instances
        """
        detections = []
        
        for pii_type in self.enabled_types:
            pattern = PIIPattern.PATTERNS.get(pii_type)
            if not pattern:
                continue
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detection = PIIDetection(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=self._calculate_confidence(pii_type, match.group())
                )
                detections.append(detection)
                self.detection_count[pii_type] += 1
        
        if detections:
            logger.warning(
                f"[PII] Detected {len(detections)} PII instances: "
                f"{[d.pii_type.value for d in detections]}"
            )
        
        return detections
    
    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate confidence score for detected PII."""
        # Credit card validation using Luhn algorithm
        if pii_type == PIIType.CREDIT_CARD:
            digits = re.sub(r'\D', '', value)
            if self._luhn_check(digits):
                return 1.0
            return 0.7
        
        # SSN validation
        if pii_type == PIIType.SSN:
            digits = re.sub(r'\D', '', value)
            if len(digits) == 9 and digits != "000000000":
                return 1.0
            return 0.6
        
        return 0.95  # Default confidence
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        
        return checksum % 10 == 0
    
    def redact(
        self,
        text: str,
        pii_types: Optional[List[PIIType]] = None,
        replacement: str = "[REDACTED]"
    ) -> Tuple[str, List[PIIDetection]]:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            pii_types: Specific PII types to redact (all if None)
            replacement: String to replace PII with
        
        Returns:
            Tuple of (redacted_text, detections)
        """
        detections = self.detect(text)
        
        # Filter by requested types
        if pii_types:
            detections = [d for d in detections if d.pii_type in pii_types]
        
        # Sort detections by position (reverse order for proper replacement)
        detections.sort(key=lambda d: d.start_pos, reverse=True)
        
        redacted_text = text
        for detection in detections:
            redacted_text = (
                redacted_text[:detection.start_pos] +
                f"{replacement}:{detection.pii_type.value}" +
                redacted_text[detection.end_pos:]
            )
        
        if detections:
            logger.info(f"[PII] Redacted {len(detections)} PII instances")
        
        return redacted_text, detections
    
    def scan_dict(
        self,
        data: Dict[str, Any],
        redact: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, List[PIIDetection]]]:
        """
        Recursively scan dictionary for PII.
        
        Args:
            data: Dictionary to scan
            redact: Whether to redact detected PII
        
        Returns:
            Tuple of (processed_data, detections_by_key)
        """
        all_detections = {}
        processed_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                detections = self.detect(value)
                if detections:
                    all_detections[key] = detections
                
                if redact and detections:
                    processed_data[key], _ = self.redact(value)
                else:
                    processed_data[key] = value
            
            elif isinstance(value, dict):
                processed_value, nested_detections = self.scan_dict(value, redact)
                processed_data[key] = processed_value
                if nested_detections:
                    all_detections[f"{key}.*"] = nested_detections
            
            elif isinstance(value, list):
                processed_list = []
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        detections = self.detect(item)
                        if detections:
                            all_detections[f"{key}[{i}]"] = detections
                        
                        if redact and detections:
                            redacted_item, _ = self.redact(item)
                            processed_list.append(redacted_item)
                        else:
                            processed_list.append(item)
                    else:
                        processed_list.append(item)
                processed_data[key] = processed_list
            
            else:
                processed_data[key] = value
        
        return processed_data, all_detections
    
    def get_statistics(self) -> Dict[str, int]:
        """Get PII detection statistics."""
        return {
            pii_type.value: count
            for pii_type, count in self.detection_count.items()
            if count > 0
        }
