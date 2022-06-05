package com.example.tiptime

import android.view.View
import android.widget.Checkable
import androidx.appcompat.widget.SwitchCompat
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.UiController
import androidx.test.espresso.ViewAction
import androidx.test.espresso.action.ViewActions
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.typeText
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.hamcrest.Matcher
import org.hamcrest.Matchers.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CalculatorTests {

    @get:Rule()
    val activity = ActivityScenarioRule(MainActivity::class.java)

    @Test
    fun calculate_20_percent_tip() {
        enterCost()
        selectOption(20)
        calculate()
        verifyTipResult("$10.00")
    }

    @Test
    fun calculate_18_percent_tip() {
        enterCost()
        selectOption(18)
        calculate()
        verifyTipResult("$9.00")
    }

    @Test
    fun calculate_15_percent_tip() {
        enterCost()
        selectOption(15)
        calculate()
        verifyTipResult("$8.00")
    }

    @Test
    fun calculate_20_percent_tip_without_round_up() {
        enterCost()
        selectOption(20)
        switchRoundUp(false)
        calculate()
        verifyTipResult("$10.00")
    }

    @Test
    fun calculate_18_percent_tip_without_round_up() {
        enterCost()
        selectOption(18)
        switchRoundUp(false)
        calculate()
        verifyTipResult("$9.00")
    }

    @Test
    fun calculate_15_percent_tip_without_round_up() {
        enterCost()
        selectOption(15)
        switchRoundUp(false)
        calculate()
        verifyTipResult("$7.50")
    }

    private fun enterCost(cost: String = "50.00") {
        onView(withId(R.id.cost_of_service_edit_text))
            .perform(typeText(cost))
            .perform(ViewActions.closeSoftKeyboard())
    }

    private fun selectOption(option: Int) {
        val optionId = when (option) {
            20 -> R.id.option_twenty_percent
            18 -> R.id.option_eighteen_percent
            15 -> R.id.option_fifteen_percent
            else -> throw IllegalArgumentException("$option not supported")
        }
        onView(withId(optionId))
            .perform(click())
    }

    private fun switchRoundUp(checked: Boolean) {
        onView(withId(R.id.round_up_switch)).perform(SwitchAction(checked))
    }

    private fun calculate() {
        onView(withId(R.id.calculate_button))
            .perform(click())
    }

    private fun verifyTipResult(expectedTip: String) {
        onView(withId(R.id.tip_result))
            .check(matches(withText(containsString(expectedTip))))
    }

    class SwitchAction(private val checked: Boolean) : ViewAction {
        override fun getConstraints(): Matcher<View> {
            return isA(SwitchCompat::class.java) as Matcher<View>
        }

        override fun getDescription(): String {
            return "switch ${if (checked) "on" else "off"}"
        }

        override fun perform(uiController: UiController?, view: View?) {
            if (view == null || !constraints.matches(view)) {
                return
            }

            if ((view as Checkable).isChecked != checked) {
                view.performClick()
            }
        }
    }
}