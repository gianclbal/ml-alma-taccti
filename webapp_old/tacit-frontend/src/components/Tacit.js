import React, { Component } from 'react';
import {
  Card,
  Container,
  Row,
  Col,
  Form,
  FormGroup,
  Button,
  Table
} from "react-bootstrap";
import Select from 'react-select';
import CSVReader from 'react-csv-reader';
import ReactWordcloud from 'react-wordcloud';
import ChartistGraph from 'react-chartist';
import { CSVLink } from "react-csv";
import sample_essays from './sample_essays.csv';

// word cloud options
const word_cloud_options = {
  colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
  enableTooltip: true,
  deterministic: false,
  fontFamily: 'Roboto',
  fontSizes: [5, 60],
  fontStyle: 'normal',
  fontWeight: 'normal',
  padding: 1,
  rotations: 4,
  rotationAngles: [0, 60],
  scale: 'sqrt',
  spiral: 'archimedean',
  transitionDuration: 1000,
};

class Tacit extends Component {
  state = {
    currentUserName: '',
    currentUserEmail: '',
    thematicCode: 0,
    uploadedData: '',
    uploadedFileName: '',
    attainmentTotalCount: 0,
    attainmentChartCounts: [0, 0],
    words: [],
    word_count: 0,
    tableData: [],
    dashboardHasData: false,
    dashboardIsLoading: false
  };

  csvReaderOptions = {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transformHeader: header =>
      header
        .toLowerCase()
        .replace(/\W/g, '_')
  }
  handleSubmit = (e) => {
    e.preventDefault()
    // set the state to loading true.
    this.setState({ dashboardIsLoading: true })
    const data = {}
    if (this.state.uploadedData && this.state.attainmentCode !== 0) {
      const { uploadedData, thematicCode } = this.state
      let paragraphs = []
      this.setState({ attainmentTotalCount: uploadedData.length })
      for (let i = 0; i < uploadedData.length; i++) {
        paragraphs.push(uploadedData[i]['sentence'])
      }
      data['csvData'] = uploadedData
      data['thematicCode'] = thematicCode
      data['essays'] = paragraphs
      // reset word cloud and table data
      this.setState({ words: [], tableData: [] })
      // Need to specify the endpoint here.
      //http://ec2-13-57-214-128.us-west-1.compute.amazonaws.com:5000
      fetch('http://127.0.0.1:5000/culturalcapitals', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      }).then((response) => {
        response.json().then((body) => {
          // Creating list for table
          //let tableList = []
          let data_table = body['data_table']
          let csv_table = body['data_table']
          csv_table.unshift(['S No.', 'Essay Id', 'CCT Present', 'Annotated Essay'])
          let total_essays = body['chart_counts'][0] + body['chart_counts'][1]

          this.setState({
            words: body['word_cloud'],
            barChartCount: {
              labels: ['Attainment', 'No Attainment'],
              series: [body['chart_counts']]
            },
            pieChartCount: {
              labels: [`CCT Present(${Math.round(body['chart_counts'][0] / total_essays * 100)}%)`, `CCT Absent(${Math.round(body['chart_counts'][1] / total_essays * 100)}%)`],
              series: body['chart_counts']
            },
            attainmentCount: body['chart_counts'][0],
            total_essays: total_essays,
            word_count: body['total_words'],
            //attainmentChartCounts: body['chart_counts'],
            tableData: data_table,
            dashboardIsLoading: false,
            csv_table: csv_table
          })
          //console.log(body)
        })
      })
    }
  }

  render() {
    return (
      <Container fluid={true}>
        <Row className="first-row">
          <Col sm="2" xs>
            <Card>
              <Card.Header><strong>CSV Data Loader</strong></Card.Header>
              <Card.Body>
                <Form onSubmit={this.handleSubmit} className="uploader-form">
                  <FormGroup>
                    <Form.Label htmlFor="thematic_code">Cultural Capital Themes (<span className="red-color-text">*required</span>)</Form.Label>
                    <Select
                      className="react-select info"
                      classNamePrefix="react-select"
                      name="thematic_code"
                      value={this.state.thematicCode}
                      onChange={value =>
                        this.setState({ thematicCode: value })
                      }
                      options={[
                        {
                          value: "",
                          label: "Select a cultural capital theme",
                          isDisabled: true
                        },
                        { value: "1", label: "Attainment" }
                      ]}
                      placeholder="Select a code"
                      required
                    />
                  </FormGroup>
                  <CSVReader
                    onFileLoaded={(data, fileInfo) => this.setState({ uploadedData: data, uploadedFileName: fileInfo })}
                    parserOptions={this.csvReaderOptions}
                    label="Upload a .csv file (*required)" />

                  <button className="btn-fill btn-primary btn upload-button" color="info" type="submit">Upload Data</button>
                </Form>
              </Card.Body>
            </Card>
          </Col>
          <Col sm="10" xs>
            <Card>
              <Card.Header><strong>Information</strong></Card.Header>
              <Card.Body>
                <ul className="cust-font-weight">
                  <li className="homepage-left-container-item3"><strong>TACIT</strong><sup>[1]</sup> is a scalable computational framework for identifying instances of
                  <strong> CCTs</strong><sup>[2]</sup>(Cultural Capital Themes) in student essays that are written as
                  part of self-affirming and reflective journaling exercises. We classify the essays based on the
                  presence or absence of cultural capital themes. Currently TACIT supports identification of
                  two types of CCTs will be included as their labeled data becomes available.</li>
                  <li className="homepage-left-container-item3"><strong>Steps:</strong>
                    <ol>
                      <li>Create a .csv file containing only the essays and their corresponding ids.
                        Sample template can be downloaded from <a className="red-color-text" href={sample_essays} target="_self" download>here</a></li>
                      <li>In the <strong>CSV Data Loader</strong> on the left, select the CCT and upload the csv file. To proceed with the analysis, click upload data button.</li>
                      <li>Annotated essays can be downloaded in the csv format from the <strong>Annotated Essays</strong> table that gets generated below.</li>
                    </ol>
                  </li>
                </ul>
                <small>
                  <ol>
                    <li>Using Text Analytics on Reflective Journaling to Identify Cultural Capitals for STEM
                      Students, 19<sup>th</sup> IEEE International Conference On Machine Learning And Applications.</li>
                    <li>T. Yosso,“Whose culture has capital? a critical race theory discussion of community
                      cultural wealth.” Race Ethnicity and Education, 8(1):69–91, 2015.</li>
                    <li>Attainment: mention of tangible goal(s) (e.g. I want to get a degree in Biology).</li>
                    <li>First-generation: mention of being the first in their family to
                      attend college (e.g. I’m the first in my family to go to college).</li>
                  </ol>
                </small>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        {this.state.dashboardIsLoading ?
          <Row>
            <Col sm="12">
              <div className="d-flex flex-column">
                <div className="p-2">
                  <br />
                  <h4 className="progress-bar-text">Processing...</h4>
                </div>
                <div className="p-8">
                  <div className="progress" style={{ 'height': '15px' }}>
                    <div className="progress-bar bg-primary progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="75" aria-valuemin="0" aria-valuemax="100" style={{ 'width': '75%' }}></div>
                  </div>
                </div>
              </div>
            </Col>
          </Row> : null
        }
        {/* Visualizations */}
        {this.state.tableData.length > 0 ?
          <Row className="mt-2">
            <Col xs sm="3">
              <Card className="card-no-shadow">
                <Card.Header className="child-card"><strong>CCT Analysis Summary</strong></Card.Header>
                <Card.Body>
                  <small>
                    <p className="analysis-card-text">Total Essays Processed: {this.state.total_essays}</p>
                    <p className="analysis-card-text">Essays Containing CCT: {this.state.attainmentCount}</p>
                  </small>
                  <hr />
                  <small><p className="analysis-card-text">The pie chart below shows the distribution of CCT across all the essays.</p></small>
                  <ChartistGraph
                    type={"Pie"}
                    data={this.state.pieChartCount}
                  />
                </Card.Body>
              </Card>

              <Card className="card-no-shadow">
                <Card.Header className="child-card"><strong>Most Frequent Words</strong></Card.Header>
                <Card.Body>
                  <small>
                    <p className="analysis-card-text">Total Unique Word Count: {this.state.word_count}</p>
                    <p className="analysis-card-text">The word cloud below is the graphical representation of <strong>word</strong> frequency in student essays,
                      in which the size of each word indicates its frequency or importance.</p>
                    <p className="analysis-card-text">The more often a specific word appears in your text, the bigger and bolder it appears in the word cloud.</p>
                  </small>
                  <hr />
                  {this.state.words.length > 0 && (
                    <ReactWordcloud options={word_cloud_options} words={this.state.words} />
                  )}
                </Card.Body>
              </Card>
            </Col>
            {/* Table */}
            <Col xs sm="9">
              {this.state.tableData.length > 0 ?

                <Card className="card-no-shadow">
                  <Card.Header className="child-card"><strong>Annotated Essays</strong> <span className="text-right download-results"><CSVLink data={this.state.csv_table} filename="annotated_essays.csv" className="btn btn-fill btn-primary">Download CSV</CSVLink></span></Card.Header>
                  <Card.Body>
                    <Table hover responsive>
                      <tbody>
                        {this.state.tableData.map((prop, idx) => {
                          return (
                            <tr key={prop[0]}>
                              <td>{prop[0]}</td>
                              <td>{prop[1]}</td>
                              <td>{prop[2]}</td>
                              <td dangerouslySetInnerHTML={{ __html: `${prop[3]}` }} />
                            </tr>
                          )
                        })}
                      </tbody>
                    </Table>
                  </Card.Body>
                </Card>


                : null}
            </Col>
          </Row>
          : null}
      </Container>
    )
  }
}

export default Tacit
