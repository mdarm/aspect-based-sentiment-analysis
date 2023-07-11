def manual_test(train_idx, test_idx):
    # Load and plit XML files.
    load_xmls('ABSA16_Restaurants_Train_SB1_v2.xml')

    # Initialize an empty DataFrame to store training data
    all_data = pd.DataFrame()

    # Process each XML file
    for index in train_idx:
        xml_file = f'part{index}.xml'
        if os.path.exists(xml_file):
            data = xml_to_dataframe(xml_file)
            all_data = pd.concat([all_data, data])

    # Train the model
    train_model(all_data)
    
    # Load the saved model and vectorizer
    model = joblib.load('trained_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Read the XML file and convert it to a DataFrame
    xml_file = f'part{test_idx}.xml'
    if os.path.exists(xml_file):
        dataframe = xml_to_dataframe(xml_file)

    # Test the model and print the F1 score
    f1 = test_model(dataframe, model, vectorizer)
    print(f'F1 score: {f1}')
