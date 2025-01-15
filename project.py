import pandas as pd
import streamlit as st
import joblib
import os

# Panel boczny do wyboru aplikacji
st.sidebar.title("Panel boczny")
selected_option = st.sidebar.radio("Wybierz aplikację", ["Strona główna", "Aplikacja"])

if selected_option == "Strona główna":
    st.title("Predyktor odejść klientów z serwisów telekomunikacyjnych")
    st.write("Witamy na naszej stronie!")

    # Zakładka z dodatkowymi opcjami
    tab1, tab2, tab3,tab6 = st.tabs(["Informacje na temat aplikacji", "Przykład działania", "Test aplikacji","Oceń Aplikacje"])

    with tab1:
        st.subheader("Opis Aplikacji")
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
                 "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
                 "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
                 "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
                 "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

        tab4, tab5 = st.tabs(["Informacje o algorytmie", "informacje o danych"])

        # Informacje o algorytmie
        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**LightGBM**")
                st.write("LightGBM (Light Gradient Boosting Machine) to zaawansowany algorytm oparty na metodzie"
                         " gradientowego wzmacniania. szczególnie wydajny przy dużych zbiorach danych.")
                st.write("W naszym modelu został skonfigurowany z parametrami: ")
                st.markdown("""
                - Pierwszy punkt
                - Drugi punkt
                - Trzeci punkt
                """)

                st.write("**Strategia poszukiwania hiperparametrów:**")
                st.markdown("""
                - Grid search z randomizacją.
                - K-fold cross-validation (5 foldów, stratified).
                - Max search time: bez limitu.
                - Random seed: 1337 dla replikacji wyników.
                """)

                st.write("**Ewaluacja i Wybór Modelu**")
                st.write("*Metody oceny wydajności modelu:*")
                st.markdown("""
                - ROC AUC: Główna metryka oceny, osiągnięto wynik 0.980.
                - Confusion matrix: Do oceny: 
                """)Do oceny dokładności (Accuracy), precyzji (Precision), czułości
                st.markdown("""
                - dokładności (Accuracy),
                - precyzji (Precision),
                - czułości     
                """)

                st.write("*Polityka podziału danych:*")
                st.markdown("""
                - Random split: Z losowym podziałem na zbiór treningowy i testowy.
                - K-fold cross-validation: Używane do dokładniejszej ewaluacji modelu.   
                """)

                st.write("**Wyniki modelowania:**")
                st.markdown("""
                • Accuracy: 95%
                • Precision: 75%
                • Recall: 100%
                • F1-Score: 86%  
                """)

            with col2:
                st.write("**Lorem ipsum**")
                st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit,"
                         " sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ")

        with tab5:

            st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit,"
                     " sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ")

            uploaded_file = st.file_uploader("Wgraj plik .csv", type="csv")

            # Check if a file has been uploaded
            if uploaded_file is not None:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(uploaded_file)

                # Display a sample of the data
                st.write("Załadowane dane:")
                st.table(df.head())

                try:
                    # Load the trained model
                    model = joblib.load("best_rf_model.pkl")  # Ensure model file is in the same directory

                    # Preprocessing the uploaded data (adjust based on `train_save_model.py`)
                    df.drop(['Age'], axis=1, inplace=True, errors='ignore')  # Drop 'Age' column if it exists
                    integer_columns = [col for col in df.columns if col not in ['Status', 'Complains', 'Churn']]
                    df[integer_columns] = df[integer_columns].astype('int')
                    df['Status'] = df['Status'].map({1: True, 2: False}).astype('bool')
                    df['Complains'] = df['Complains'].map({1: True, 0: False}).astype('bool')

                    # Drop the target column if it exists
                    X_infer = df.drop(['Churn'], axis=1, errors='ignore')

                    # Make predictions
                    predictions = model.predict(X_infer)

                    # Add predictions to the DataFrame
                    df['Prediction'] = predictions

                    # Display rows incrementally with session state tracking
                    if "rows_shown" not in st.session_state:
                        st.session_state.rows_shown = 50

                    rows_shown = st.session_state.rows_shown
                    st.write(f"Wyświetlono {rows_shown} wierszy z danymi i predykcjami:")
                    st.table(df.iloc[:rows_shown])

                    if rows_shown < len(df):
                        if st.button(f"Pokaż kolejne {min(100, len(df) - rows_shown)} wierszy",
                                     key=f"button_{rows_shown}"):
                            st.session_state.rows_shown += 100

                    # Add a download button for the predictions CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Pobierz dane z predykcjami jako CSV",
                        data=csv,
                        file_name="predykcje.csv",
                        mime="text/csv",
                    )

                except FileNotFoundError:
                    st.error("Plik modelu `best_rf_model.pkl` nie został znaleziony. Upewnij się, że znajduje się w tym samym folderze co ten skrypt.")
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas przetwarzania: {e}")
    with tab6:
        st.subheader("Oceń Aplikacje")
        st.write("Twoja opinia jest dla nas ważna! Oceń aplikację i podziel się swoimi uwagami.")
        st.write("Oceń zadowolenie aplikacji")
        rating=st.slider("Zadowolenie 0-10",0,10,5)
        feedback = st.text_area("Podziel się swoją opinią (opcjonalne):")
        if st.button("Prześlij opinię"):
            st.success("Dziękujemy za Twoją opinię!")