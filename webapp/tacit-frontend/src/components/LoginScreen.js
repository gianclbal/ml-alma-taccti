import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import background_image from './../images/banner.png';
import tacit_logo from './../images/tacitlogo.png';

import {
  Card,
  Container,
  Row,
  Col
} from "react-bootstrap";

const LoginScreen = () => {
  const { loginWithRedirect, isAuthenticated } = useAuth0();

  return (
    !isAuthenticated && (
      <div>
        <Container fluid={true}>
          <Row className="mt-4">
            <Col md="6" xs>
              <img className="img-fluid homepage-banner" src={background_image} alt="" />
            </Col>
            <Col md="6" xs>
              <Card>
                <Card.Body>
                  <Row>
                    <div className="col">
                      <div className="d-flex align-items-center flex-column">
                        <div className="p-3">
                          <img src={tacit_logo} alt="logoname" className="img-fluid" />
                        </div>
                        <div className="p-1">
                          <p className="homepage-left-container-item2">The <strong className="extra-bold-element">ALMA Project </strong>(meaning heart/soul in Spanish)
                          focuses on retention of Historically Underrepresented (HU) students in Science,
                          Technology, Engineering and Mathematics (STEM) fields by affirming their
                          cultural capital - the skills, knowledge, communities, and abilities that HU students
                          possess. As part of the Alma program students are asked to reflect and respond to essay prompts such
                          as ‘why am I here?’, ‘what do I do when life gets challenging’ – prompts that are designed to
                          help students remember and assert their strengths, career motivations, life purpose and experience. Alma team augmented the <i>community wealth model</i> proposed by <strong>Tarro J Yasso</strong> with
                          additional CCTs to capture additional themes appearing in the student essays in STEM courses. Some of the CCTs are:</p>
                          <ul className="homepage-left-container-item2">
                            <li><strong className="extra-bold-element">Attainment</strong>: mention of tangible goal(s) (e.g. I want to get a degree in Biology).</li>
                            <li><strong className="extra-bold-element">Community Consciousness</strong>: mention of solidarity with community and the desire to give
                              back to a community one identifies as being part of (e.g. I want to help my people).</li>
                            <li><strong className="extra-bold-element">Familial</strong>: mention of support provided by family, whether tangible
                              support (e.g. food, financial support), emotional support, or role modeling.</li>
                            <li><strong className="extra-bold-element">First Generation</strong>: mention of being the first in their family to
                              attend college (e.g. I’m the first in my family to go to college).</li>
                          </ul>
                        </div>
                        <div className="p-1">
                          <p className="homepage-left-container-item"><strong className="extra-bold-element">TACIT:</strong> (<strong className="extra-bold-element">T</strong>hem<strong className="extra-bold-element">A</strong>tic <strong className="extra-bold-element">C</strong>oding tool to <strong className="extra-bold-element">I</strong>dentify cultural capital <strong className="extra-bold-element">T</strong>hemes)
                          is a scalable computational framework for identifying instances of cultural capital themes in student essays that are written as part of self-affirming and reflective
                           journaling exercises. Currently TACIT supports identification of two types of CCTs (<strong className="extra-bold-element">Attainment </strong> 
                             and <strong className="extra-bold-element">First-generation</strong>) but the framework is designed to easily onboard additional CCTs as their labeled data becomes available.</p>
                        </div>
                        <div className="p-1">
                          <button className="btn btn-fill btn-block btn-primary" onClick={() => loginWithRedirect()}>Login/Sign Up</button>
                        </div>
                      </div>
                    </div>
                  </Row>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Container>
      </div>
    )
  )
}

export default LoginScreen