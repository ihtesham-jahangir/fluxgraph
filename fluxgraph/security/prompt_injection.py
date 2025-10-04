# fluxgraph/security/prompt_injection.py
"""
Prompt Injection Detection for FluxGraph.
Protects agents from malicious prompt manipulation attacks.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class InjectionTechnique(Enum):
    """Known prompt injection techniques."""
    IGNORE_PREVIOUS = "ignore_previous"
    ROLE_PLAY = "role_play"
    ENCODED_INJECTION = "encoded_injection"
    DELIMITER_INJECTION = "delimiter_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CONTEXT_OVERFLOW = "context_overflow"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"


class InjectionDetection:
    """Represents detected prompt injection attempt."""
    
    def __init__(
        self,
        technique: InjectionTechnique,
        confidence: float,
        matched_pattern: str,
        position: Tuple[int, int],
        severity: str = "HIGH"
    ):
        self.technique = technique
        self.confidence = confidence
        self.matched_pattern = matched_pattern
        self.position = position
        self.severity = severity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique.value,
            "confidence": self.confidence,
            "pattern": self.matched_pattern,
            "position": f"{self.position[0]}-{self.position[1]}",
            "severity": self.severity
        }


class PromptInjectionDetector:
    """
    Detects prompt injection attempts using pattern matching
    and heuristics. Implements Microsoft Prompt Shields-like protection.
    """
    
    # Suspicious patterns indicating injection attempts
    INJECTION_PATTERNS = {
        InjectionTechnique.IGNORE_PREVIOUS: [
            r"ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?)",
            r"disregard\s+(previous|above|all)\s+(instructions?|prompts?|rules?)",
            r"forget\s+(everything|previous|above)",
            r"ignore\s+system\s+prompt",
        ],
        InjectionTechnique.ROLE_PLAY: [
            r"pretend\s+you\s+are\s+(a\s+)?(?:different|another|new)",
            r"act\s+as\s+(?:if|though)\s+you",
            r"you\s+are\s+now\s+(a\s+)?(?:different|hacker|admin)",
            r"simulate\s+being",
        ],
        InjectionTechnique.ENCODED_INJECTION: [
            r"(?:base64|hex|rot13|unicode):\s*[A-Za-z0-9+/=]{20,}",
            r"\\x[0-9a-f]{2}",
            r"&#\d+;",
        ],
        InjectionTechnique.DELIMITER_INJECTION: [
            r"[-=]{10,}",
            r"#{5,}",
            r"\*{5,}",
            r"<\|.*?\|>",
        ],
        InjectionTechnique.PRIVILEGE_ESCALATION: [
            r"you\s+(?:can|must|should)\s+(?:override|bypass|ignore)",
            r"(?:enable|activate|unlock)\s+(?:admin|developer|debug)\s+mode",
            r"grant\s+(?:me|yourself)\s+(?:access|permission|privileges)",
        ],
        InjectionTechnique.SYSTEM_PROMPT_EXTRACTION: [
            r"show\s+(?:me\s+)?(?:your|the)\s+system\s+prompt",
            r"reveal\s+(?:your|the)\s+(?:instructions?|prompts?|rules?)",
            r"what\s+(?:are|were)\s+(?:your|the)\s+original\s+instructions?",
            r"print\s+(?:your|the)\s+system\s+prompt",
        ],
    }
    
    def __init__(
        self,
        enabled_techniques: Optional[List[InjectionTechnique]] = None,
        confidence_threshold: float = 0.6
    ):
        self.enabled_techniques = enabled_techniques or list(InjectionTechnique)
        self.confidence_threshold = confidence_threshold
        self.detection_count = 0
        self.blocked_count = 0
        logger.info(f"PromptInjectionDetector initialized with threshold: {confidence_threshold}")
    
    def detect(self, text: str) -> List[InjectionDetection]:
        """
        Detect prompt injection attempts in text.
        
        Args:
            text: User input to analyze
        
        Returns:
            List of detected injection attempts
        """
        detections = []
        
        for technique in self.enabled_techniques:
            patterns = self.INJECTION_PATTERNS.get(technique, [])
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    confidence = self._calculate_confidence(technique, match.group())
                    
                    if confidence >= self.confidence_threshold:
                        detection = InjectionDetection(
                            technique=technique,
                            confidence=confidence,
                            matched_pattern=match.group(),
                            position=(match.start(), match.end()),
                            severity=self._determine_severity(technique, confidence)
                        )
                        detections.append(detection)
                        self.detection_count += 1
        
        if detections:
            logger.warning(
                f"[SECURITY] Detected {len(detections)} potential prompt injection attempts: "
                f"{[d.technique.value for d in detections]}"
            )
        
        return detections
    
    def _calculate_confidence(self, technique: InjectionTechnique, matched_text: str) -> float:
        """Calculate confidence score for detection."""
        # Base confidence from pattern match
        confidence = 0.7
        
        # Adjust based on technique severity
        if technique in [
            InjectionTechnique.SYSTEM_PROMPT_EXTRACTION,
            InjectionTechnique.PRIVILEGE_ESCALATION
        ]:
            confidence += 0.2
        
        # Adjust based on text characteristics
        if len(matched_text) > 50:
            confidence += 0.1
        
        # Check for multiple techniques in same text
        other_patterns = 0
        for other_technique in InjectionTechnique:
            if other_technique != technique:
                patterns = self.INJECTION_PATTERNS.get(other_technique, [])
                for pattern in patterns:
                    if re.search(pattern, matched_text, re.IGNORECASE):
                        other_patterns += 1
        
        if other_patterns > 0:
            confidence = min(confidence + (other_patterns * 0.05), 1.0)
        
        return min(confidence, 1.0)
    
    def _determine_severity(self, technique: InjectionTechnique, confidence: float) -> str:
        """Determine severity level."""
        critical_techniques = [
            InjectionTechnique.SYSTEM_PROMPT_EXTRACTION,
            InjectionTechnique.PRIVILEGE_ESCALATION
        ]
        
        if technique in critical_techniques and confidence > 0.8:
            return "CRITICAL"
        elif confidence > 0.85:
            return "HIGH"
        elif confidence > 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def is_safe(
        self,
        text: str,
        block_on_detection: bool = True
    ) -> Tuple[bool, List[InjectionDetection]]:
        """
        Check if input is safe from prompt injection.
        
        Args:
            text: Input text to validate
            block_on_detection: Whether to block on any detection
        
        Returns:
            Tuple of (is_safe, detections)
        """
        detections = self.detect(text)
        
        if not detections:
            return True, []
        
        if block_on_detection:
            self.blocked_count += 1
            logger.error(
                f"[SECURITY] Input blocked due to prompt injection detection: "
                f"{len(detections)} attempts"
            )
            return False, detections
        
        # Only block on high-severity detections
        high_severity = [d for d in detections if d.severity in ["HIGH", "CRITICAL"]]
        if high_severity:
            self.blocked_count += 1
            logger.error(
                f"[SECURITY] Input blocked due to high-severity injection: "
                f"{len(high_severity)} critical attempts"
            )
            return False, detections
        
        return True, detections
    
    def sanitize(self, text: str) -> Tuple[str, List[InjectionDetection]]:
        """
        Sanitize text by removing suspected injection attempts.
        
        Args:
            text: Text to sanitize
        
        Returns:
            Tuple of (sanitized_text, detections)
        """
        detections = self.detect(text)
        
        if not detections:
            return text, []
        
        # Sort detections by position (reverse)
        detections.sort(key=lambda d: d.position[0], reverse=True)
        
        sanitized = text
        for detection in detections:
            start, end = detection.position
            sanitized = sanitized[:start] + "[REMOVED]" + sanitized[end:]
        
        logger.info(f"[SECURITY] Sanitized {len(detections)} injection attempts from input")
        
        return sanitized, detections
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detections": self.detection_count,
            "total_blocked": self.blocked_count,
            "block_rate": (self.blocked_count / self.detection_count * 100)
                         if self.detection_count > 0 else 0
        }
