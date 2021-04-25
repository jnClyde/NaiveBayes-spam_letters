from NaiveBayes import *
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from termcolor import cprint
import csv

default_key_words = ['100', '%', 'миллион', 'тысяча', 'скрытый', 'успех', 'новинка', 'реклама', 'скидка', 'спам',
                     'распродажа', 'похудение', 'секс', '18+', '21+', 'срочно', 'выгода', 'гарантия',
                     'победа', 'конкурс', 'выигрыш', 'приз', 'бесплатно', 'быстро', 'позвонить', 'только', 'сегодня',
                     'низкий', 'цена', 'розыгрыш', 'похудеть']


# ----------------------------------------------------------------------------------------------------------------------
# Записывает передеанные слова в формате csv в файл key_words.txt
# ----------------------------------------------------------------------------------------------------------------------
def set_key_words(words=None):
    if words is None:
        words = default_key_words
    stemmer = SnowballStemmer("russian")
    with open('data/key_words.txt', 'w') as file:
        for word in words:
            file.write(stemmer.stem(word) + '\n')


# ----------------------------------------------------------------------------------------------------------------------
# @return       -- Возвращает массив слов из файла key_words.txt
# ----------------------------------------------------------------------------------------------------------------------
def get_key_words():
    with open('data/key_words.txt', 'r') as file:
        words = file.readlines()
    words = [w.replace('\n', '') for w in words]
    return words


# ----------------------------------------------------------------------------------------------------------------------
# Возвращает обучающие данные из файла train_data.csv в их исходном виде
# ----------------------------------------------------------------------------------------------------------------------
def get_train_data():
    data = []
    with open('data/train_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


# ----------------------------------------------------------------------------------------------------------------------
# Разбивает текст на леммы
# ----------------------------------------------------------------------------------------------------------------------
def lemmatize_train_data(data):
    stemmer = SnowballStemmer("russian")
    data = [[word_tokenize(w[0], language='russian'), w[1]] for w in data]
    result = []
    for item in data:
        current_result = [stemmer.stem(w) for w in item[0]]
        result.append([current_result, item[1]])
    return result


def print_result(data, text):
    count = 0
    for item in data:
        print(text[count])
        count += 1
        if item == 0:
            cprint('Не спам', 'green')
        else:
            cprint('Спам', 'red')
        print_sep()


def print_sep():
    for i in range(0, 150):
        print('-', end='')
    print('\n')


# ----------------------------------------------------------------------------------------------------------------------
# Переводит текст в векторное представление
# ----------------------------------------------------------------------------------------------------------------------
def get_text_vector(text_lemmas, key_words):
    vec = []
    stemmer = SnowballStemmer("russian")
    text_lemmas_t = [stemmer.stem(w) for w in text_lemmas]
    for key_word in key_words:
        vec.append(text_lemmas_t.count(key_word))
    return vec


def main():
    set_key_words()
    # Работа с данными
    data = lemmatize_train_data(get_train_data())
    key_words = get_key_words()
    text_vector = []
    text_result = []
    for d in data:
        text_vector.append(get_text_vector(d[0], key_words))
        text_result.append(int(d[1]))

    # Запуск и обучение модели
    model = BayesClassifier()
    model.set_xy(text_vector, text_result)
    model.learn()

    print('Хотите проверить собственное(ые) письмо(а) или посмотреть пример работы программы?(Ответ: 1 или 2)')
    answer = input()

    if int(answer) != 1:
        # пример
        test_texts = ['Не упустите самые главные акции января! Скидки до 50% на горные лыжи и сноуборд. Сделайте эту зиму '
                      'запоминающейся!',
                      'ДРУЗЬЯ, стартовала регистрация на VII Всероссийский онлайн-чемпионат «Изучи интернет – управляй им» '
                      'в командном и индивидуальном зачетах!']
        test_text_vecs = [get_text_vector(word_tokenize(test_text, language='russian'), key_words) for test_text in test_texts]

        print_result(model.predict(test_text_vecs), test_texts)

    else:
        print('Хотите проверить одно письмо или несколько писем?(Ответ: 1 или 2)')
        answer = input()

        mail_texts = ''
        if int(answer) == 1:
            print('Введите текст письма: ')
            mail_text = input()
            mail_texts = [mail_text]
            print()

        else:
            print('Введите путь к файлу, где хранятся тексты писем: ')
            path = input()
            print()
            file = open(path, encoding='utf-8')
            mail_texts = file.readlines()

        mail_text_vecs = [get_text_vector(word_tokenize(mail_text, language='russian'), key_words) for mail_text in
                          mail_texts]
        print_result(model.predict(mail_text_vecs), mail_texts)


if __name__ == '__main__':
    main()
