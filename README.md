# ADM-HW3

This repository includes the solutions of Homework 3 from the Group 2:

<div style="float: left;">
    <table>
        <tr>
            <th>Student</th>
            <th>GitHub</th>
            <th>Matricola</th>
            <th>E-Mail</th>
        </tr>
        <tr>
            <td>Gloria Kim</td>
            <td>keemgloria</td>
            <td>1862339</td>
            <td>kim.1862339@studenti.uniroma1.it</td>
        </tr>
        <tr>
            <td>Tim Ragno</td>
            <td>griimish</td>
            <td>2116901</td>
            <td>ragno.2116901@studenti.uniroma1.it</td>
        </tr>
        <tr>
            <td>Arash Bakhshaee Babaroud</td>
            <td>ArashB1230</td>
            <td>2105709</td>
            <td>Arashbakhshaee@gmail.com</td>
        </tr>
        <tr>
            <td>André Leibrant</td>
            <td>JesterProphet</td>
            <td>2085698</td>
            <td>leibrant.2085698@studenti.uniroma1.it</td>
        </tr>
    </table>
</div>

The main notebook is `main.ipynb` (HTML version: `main.html`) which reads the module `engine.py`. In addition, we outsourced the functions for the crawling in `crawler.py`, the parsing in `parser.py`, and the preprocessing in `preprocess.py`. 

The image `cmd.png` is the screenshot of the command line output. 

The created map for question 4 is stored as `map.html`.

The shell script for the Command Line Question is `CommandLine.sh`.

**Note:**

- We directly executed the scripts `crawler.py`, `parser.py`, and `preprocess.py` in advance inside the notebook using the `subprocess` library.
- We created all the inverted indexes using functions stored inside `engine.py` which we then stored locally inside pickle files and loaded them for our engines later.
- We excluded for privacy reasons the real Google Maps API Key.
- We used the provided `kepler_config.json` to create our map in question 4.


