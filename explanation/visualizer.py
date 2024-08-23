from bs4 import BeautifulSoup
import torch
from captum.attr import visualization as viz
class Visualizer:
    @staticmethod
    def visualize_attributions(model, tokenizer, attributions, tokens, delta, text, label, html_output_path):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        
        token_words = tokenizer.convert_ids_to_tokens(tokens)
        
        input_ids = torch.tensor([tokens]).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        pred_prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(pred_prob).item()
        
        vis_data_records = []
        vis_data_records.append(viz.VisualizationDataRecord(
                                attributions,
                                pred_prob[0, pred_class].item(),
                                pred_class,
                                label,
                                "label",
                                attributions.sum(),
                                token_words,
                                delta))
        
        html_content = viz.visualize_text(vis_data_records).data
        # Save HTML to file
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(str(BeautifulSoup(html_content, "html.parser").prettify()))

        return html_output_path
