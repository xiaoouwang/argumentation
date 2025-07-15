import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, Dict, Any, List


class ArgumentationTransformer(nn.Module):
    """
    Transformer-based model for argumentation mining tasks.
    This model can handle both argument component identification
    and argument relation classification.
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        num_labels: int = 5,
        dropout_prob: float = 0.1,
        relation_types: int = 3,
        enable_relation_classification: bool = True
    ):
        """
        Initialize the model.

        Args:
            model_name_or_path: Name or path of the pretrained transformer model
            num_labels: Number of argument component labels
            dropout_prob: Dropout probability
            relation_types: Number of relation types for relation classification
            enable_relation_classification: Whether to enable relation classification
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        # Component classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.component_classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Relation classification head (optional)
        self.enable_relation_classification = enable_relation_classification
        if enable_relation_classification:
            self.relation_classifier = nn.Linear(self.config.hidden_size * 2, relation_types)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        component_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None,
        relation_pairs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Token input IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (only for BERT-like models)
            component_labels: Ground truth labels for component classification
            relation_labels: Ground truth labels for relation classification
            relation_pairs: Pairs of component indices for relation classification

        Returns:
            Dictionary with loss and logits
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Component classification
        pooled_output = self.dropout(pooled_output)
        component_logits = self.component_classifier(pooled_output)

        result = {"component_logits": component_logits}

        # Calculate loss if labels are provided
        loss = 0
        if component_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            component_loss = loss_fct(component_logits.view(-1, self.component_classifier.out_features),
                                     component_labels.view(-1))
            loss += component_loss
            result["component_loss"] = component_loss

        # Relation classification (if enabled and relation pairs are provided)
        if self.enable_relation_classification and relation_pairs is not None:
            batch_size, pair_count, _ = relation_pairs.size()
            relation_representations = []

            # Extract token representations for each component in the pair
            for b in range(batch_size):
                pairs = relation_pairs[b]  # shape: [pair_count, 2]
                batch_representations = []

                for p in range(pair_count):
                    arg1_idx, arg2_idx = pairs[p]
                    # Skip invalid pairs (padding)
                    if arg1_idx == -1 or arg2_idx == -1:
                        # Use zero vector for padding
                        batch_representations.append(torch.zeros(self.config.hidden_size * 2,
                                                               device=sequence_output.device))
                        continue

                    # Get representations for both arguments
                    arg1_repr = sequence_output[b, arg1_idx]
                    arg2_repr = sequence_output[b, arg2_idx]

                    # Concatenate representations
                    pair_repr = torch.cat([arg1_repr, arg2_repr], dim=-1)
                    batch_representations.append(pair_repr)

                # Stack all pair representations for this batch
                batch_tensor = torch.stack(batch_representations, dim=0)
                relation_representations.append(batch_tensor)

            # Stack all batch representations
            relation_repr = torch.stack(relation_representations, dim=0)
            relation_repr = self.dropout(relation_repr)

            # Classify relations
            relation_logits = self.relation_classifier(relation_repr)
            result["relation_logits"] = relation_logits

            # Calculate relation loss if labels are provided
            if relation_labels is not None:
                relation_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                relation_loss = relation_loss_fct(relation_logits.view(-1, self.relation_classifier.out_features),
                                               relation_labels.view(-1))
                loss += relation_loss
                result["relation_loss"] = relation_loss

        result["loss"] = loss
        return result

    def save_pretrained(self, output_dir: str) -> None:
        """Save the model to the output directory."""
        self.transformer.save_pretrained(output_dir)
        torch.save(self.state_dict(), f"{output_dir}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "ArgumentationTransformer":
        """Load the model from the saved path."""
        model = cls(model_name_or_path=model_path, **kwargs)
        model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
        return model


class ArgumentComponentDetector:
    """
    Wrapper class for using the ArgumentationTransformer model
    for argument component detection.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        id2label: Optional[Dict[int, str]] = None
    ):
        self.device = device
        self.model = ArgumentationTransformer.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        if id2label is None:
            # Default mapping
            self.id2label = {
                0: "Claim",
                1: "Premise",
                2: "Major_Claim",
                3: "Non_argumentative",
                4: "Backing"
            }
        else:
            self.id2label = id2label

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict argument components for a list of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of dictionaries with predicted labels and confidence scores
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        component_logits = outputs["component_logits"]
        component_probs = torch.softmax(component_logits, dim=-1)
        component_predictions = torch.argmax(component_probs, dim=-1)

        results = []
        for idx, (text, pred, probs) in enumerate(zip(texts, component_predictions, component_probs)):
            label = self.id2label[pred.item()]
            confidence = probs[pred].item()

            results.append({
                "text": text,
                "predicted_label": label,
                "confidence": confidence,
                "probabilities": {self.id2label[i]: p.item() for i, p in enumerate(probs)}
            })

        return results