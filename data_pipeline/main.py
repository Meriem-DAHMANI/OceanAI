import pandas as pd
from datasets import Dataset, DatasetDict
from  web_scraping import get_category_links, get_category_content
from clean_data import clean_text, data_split
from build_sft_dataset import generate_qa_dataset

def run_pipeline():
    categories = [
    "Marine_biology",
    "Marine_ecology",
    "Ocean",
    "Coral_reefs",
    "Marine_mammals",
    "Oceanography",
    "Fisheries_science",
    "Marine_conservation",
    ]

    # get marine URLS 
    marine_urls = []
    for marine_category in categories : 
        marine_urls += get_category_links(marine_category) 
    
    # get content 
    marine_data = get_category_content(marine_urls)
    print(f"Total articles: {len(marine_data)}")

    # Convert the list of dicts into a Dataset
    # 1. Make a DataFrame – pandas does the heavy lifting
    df = pd.DataFrame(marine_data)

    # 2. Turn the DataFrame into a Hugging Face Dataset
    marine_dataset = Dataset.from_pandas(df)

    # clean the dataset 
    marine_dataset = marine_dataset.map(lambda ex: {"clean_content": clean_text(ex["content"])},
                        remove_columns=["content"])
    
    # split dataset for train /test for SFT/CPT
    pretrain_train, pretrain_test, finetune_train, finetune_test = data_split(marine_dataset)

    # build SFT (Q/A) dataset
    qa_train = generate_qa_dataset(finetune_train)
    qa_test  = generate_qa_dataset(finetune_test)
    print(f"Q/A train pairs: {len(qa_train)}")
    print(f"Q/A test pairs:  {len(qa_test)}")

    ## CPT dataset 
    cpt_dataset = DatasetDict({
        "train": pretrain_train.rename_column("clean_content", "text"),
        "test":  pretrain_test.rename_column("clean_content", "text")
    })



if __name__ == "__main__":
    run_pipeline()

